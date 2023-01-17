import tensorflow as tf

import math, re, os

import numpy as np

import pandas as pd



from matplotlib import pyplot as plt

from kaggle_datasets import KaggleDatasets

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

print("Tensorflow version " + tf.__version__)



from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.applications import *

from keras import backend

AUTO = tf.data.experimental.AUTOTUNE



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import DataLoader



from torchvision import datasets

from torchvision import transforms



import matplotlib.pyplot as plt

from PIL import Image



if torch.cuda.is_available():

    torch.backends.cudnn.deterministic = True

    

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



tpu = None

if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.



print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
IMAGE_SIZE = [512, 512] # at this size, a GPU will run out of memory. Use the TPU

EPOCHS = 12

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}

GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition



#Various settings :     

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')

VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



# Took this from one of the notebooks in the competition:

SKIP_VALIDATION = False

if SKIP_VALIDATION:

    TRAINING_FILENAMES = TRAINING_FILENAMES + VALIDATION_FILENAMES

    

SKIP_VISUALISATION = True



CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09

           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19

           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29

           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39

           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49

           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59

           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69

           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79

           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89

           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                      



# Function which transforms a batch of images into numpy structures

def batch_to_numpy_images_and_labels(data):

    images, labels = data

    numpy_images = images.numpy()

    numpy_labels = labels.numpy()

    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings

        numpy_labels = [None for _ in enumerate(numpy_images)]

    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels



# Function which fetches the batched dataset : 

def get_training_dataset():

    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(ordered=False):

    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def weighted_categorical_crossentropy(weights):

    """

    A weighted version of keras.objectives.categorical_crossentropy

    

    Variables:

        weights: numpy array of shape (C,) where C is the number of classes

    

    Usage:

        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.

        loss = weighted_categorical_crossentropy(weights)

        model.compile(loss=loss,optimizer='adam')

    """

    

    weights = K.variable(weights)

        

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32) # add Version 2

        y_pred = tf.cast(y_pred, tf.float32) # add Version 2

        # scale predictions so that the class probas of each sample sum to 1

        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

        # clip to prevent NaN's and Inf's

        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())

        # calc

        loss = y_true * K.log(y_pred) * weights

        loss = -K.sum(loss, -1)

        return loss

    

    return loss



def generate_nb_samples_per_class(CLASSES, NUM_TRAINING_IMAGES, train_batch):

    # Create an empty dict in which will be stored the number of samples per class

    n = np.zeros(len(CLASSES), dtype='float')

    names = list(range(len(CLASSES)))

    samples_per_classes = dict(zip(names, n))



    # This is the only solution I found to stop the loop:

    count = 0

    while(count < NUM_TRAINING_IMAGES):

        count = count + training_batch_size

        im, np_labels = batch_to_numpy_images_and_labels(next(train_batch))

    

        # get the labels in batch, count the samples per class and add to the dict :

        unique, counts = np.unique(np_labels, return_counts=True)

        count_per_label = np.asarray((unique, counts)).T

    

        for row in count_per_label:

            samples_per_classes[row[0]] = samples_per_classes[row[0]] + row[1]

    

    return samples_per_classes

# Run the function, get the training dataset : 

TRAINING_BATCH_SIZE = 200

training_dataset = get_training_dataset()

training_dataset = training_dataset.unbatch().batch(TRAINING_BATCH_SIZE)

train_batch = iter(training_dataset)
# samples_per_classes = generate_nb_samples_per_class(CLASSES, NUM_TRAINING_IMAGES, train_batch)
SAMPLES_PER_CLASSES_TRAINING_VALIDATION = {0: 356.0, 1: 33.0, 2: 27.0, 3: 31.0, 4: 921.0, 5: 112.0, 6: 23.0, 7: 136.0, 8: 116.0, 9: 110.0, 10: 182.0, 11: 53.0, 12: 114.0, 13: 340.0, 14: 300.0, 15: 28.0, 16: 69.0, 17: 69.0, 18: 112.0, 19: 34.0, 20: 24.0, 21: 130.0, 22: 63.0, 23: 25.0, 24: 106.0, 25: 111.0, 26: 28.0, 27: 47.0, 28: 150.0, 29: 143.0, 30: 138.0, 31: 30.0, 32: 29.0, 33: 24.0, 34: 20.0, 35: 46.0, 36: 77.0, 37: 33.0, 38: 27.0, 39: 93.0, 40: 82.0, 41: 128.0, 42: 83.0, 43: 136.0, 44: 25.0, 45: 222.0, 46: 162.0, 47: 343.0, 48: 565.0, 49: 730.0, 50: 263.0, 51: 137.0, 52: 147.0, 53: 610.0, 54: 45.0, 55: 72.0, 56: 117.0, 57: 79.0, 58: 48.0, 59: 77.0, 60: 35.0, 61: 35.0, 62: 123.0, 63: 39.0, 64: 73.0, 65: 42.0, 66: 28.0, 67: 1001.0, 68: 339.0, 69: 125.0, 70: 129.0, 71: 179.0, 72: 218.0, 73: 601.0, 74: 165.0, 75: 397.0, 76: 159.0, 77: 179.0, 78: 115.0, 79: 153.0, 80: 196.0, 81: 123.0, 82: 173.0, 83: 147.0, 84: 39.0, 85: 38.0, 86: 157.0, 87: 196.0, 88: 125.0, 89: 65.0, 90: 130.0, 91: 149.0, 92: 32.0, 93: 178.0, 94: 161.0, 95: 165.0, 96: 130.0, 97: 53.0, 98: 47.0, 99: 31.0, 100: 40.0, 101: 32.0, 102: 503.0, 103: 974.0}



SAMPLES_PER_CLASSES_TRAINING = {0: 271.0, 1: 25.0, 2: 17.0, 3: 22.0, 4: 717.0, 5: 87.0, 6: 16.0, 7: 98.0, 8: 87.0, 9: 86.0, 10: 137.0, 11: 47.0, 12: 91.0, 13: 270.0, 14: 224.0, 15: 22.0, 16: 54.0, 17: 51.0, 18: 94.0, 19: 27.0, 20: 20.0, 21: 104.0, 22: 50.0, 23: 21.0, 24: 85.0, 25: 84.0, 26: 22.0, 27: 33.0, 28: 114.0, 29: 108.0, 30: 106.0, 31: 25.0, 32: 23.0, 33: 20.0, 34: 18.0, 35: 37.0, 36: 55.0, 37: 24.0, 38: 19.0, 39: 76.0, 40: 64.0, 41: 92.0, 42: 61.0, 43: 109.0, 44: 20.0, 45: 178.0, 46: 126.0, 47: 264.0, 48: 426.0, 49: 554.0, 50: 195.0, 51: 96.0, 52: 114.0, 53: 468.0, 54: 37.0, 55: 60.0, 56: 100.0, 57: 62.0, 58: 37.0, 59: 60.0, 60: 28.0, 61: 27.0, 62: 100.0, 63: 29.0, 64: 55.0, 65: 33.0, 66: 20.0, 67: 776.0, 68: 258.0, 69: 97.0, 70: 105.0, 71: 141.0, 72: 170.0, 73: 458.0, 74: 126.0, 75: 305.0, 76: 118.0, 77: 139.0, 78: 87.0, 79: 123.0, 80: 154.0, 81: 97.0, 82: 132.0, 83: 114.0, 84: 28.0, 85: 31.0, 86: 124.0, 87: 149.0, 88: 92.0, 89: 48.0, 90: 106.0, 91: 109.0, 92: 24.0, 93: 138.0, 94: 127.0, 95: 133.0, 96: 101.0, 97: 41.0, 98: 33.0, 99: 23.0, 100: 33.0, 101: 26.0, 102: 398.0, 103: 734.0}
with strategy.scope():

    samples_per_classes = SAMPLES_PER_CLASSES_TRAINING

    if SKIP_VALIDATION:

        samples_per_classes = SAMPLES_PER_CLASSES_TRAINING_VALIDATION

    

    df_weights = pd.DataFrame.from_dict(samples_per_classes, orient='index', columns=['nb_samples'])



    # Ballance weights for each class:  

    sample_numbers = df_weights[df_weights['nb_samples']>0].to_numpy().flatten()

    max_samples = np.max(sample_numbers)

    df_weights = df_weights.applymap(lambda x: max_samples/float(x) if x>0 else 0.0)

    weights = df_weights.to_numpy().T



    # And finally :

    home_made_loss = weighted_categorical_crossentropy(weights)

    

    # Let us pick this one, as an example : 

    pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    pretrained_model.trainable = False # tramsfer learning

    

    # original model

    model_original = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len(CLASSES), activation='softmax')

    ])

    

    model.compile(

        optimizer='adam',

        loss = home_made_loss,

        metrics=['sparse_categorical_accuracy']

    )
history = model.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=10, callbacks=[lr_callback], validation_data=get_validation_dataset())