import random

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

import tensorflow as tf

# As TPUs require acces to the GCS path

from kaggle_datasets import KaggleDatasets

!pip install typeguard

!pip install -q --no-deps tensorflow-addons~=0.7

import tensorflow_addons as tfa

print('Tensorflow version ' + tf.__version__)

from sklearn.model_selection import KFold
AUTO = tf.data.experimental.AUTOTUNE



# Cluster Resolver for Google Cloud TPUs.

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

# Connects to the given cluster.

tf.config.experimental_connect_to_cluster(tpu)

# Initialize the TPU devices.

tf.tpu.experimental.initialize_tpu_system(tpu)

# TPU distribution strategy implementation.

strategy = tf.distribute.experimental.TPUStrategy(tpu)



# Configuration

IMAGE_SIZE = [512, 512]

EPOCHS = 50

BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512, 512]
# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()



GCS_PATH_SELECT = { # available image sizes

    192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',

    224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',

    331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',

    512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'

}



GCS_PATH = GCS_PATH_SELECT[IMAGE_SIZE[0]]



TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec') + tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')

TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') # predictions on this dataset should be submitted for the competition
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

           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102
# Learning rate schedule for TPU, GPU and CPU.

# Using an LR ramp up because fine-tuning a pre-trained model.

# Starting with a high LR would break the pre-trained weights.



LR_START = 0.00001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.00001

LR_RAMPUP_EPOCHS = 5

LR_SUSTAIN_EPOCHS = 0

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose = True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

plt.plot(rng, y)

print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['id']

    return image, idnum # returns a dataset of image(s)



def load_dataset(filenames, labeled = True, ordered = False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # Diregarding data order. Order does not matter since we will be shuffling the data anyway

    

    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed

        

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # use data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls = AUTO) # returns a dataset of (image, label) pairs if labeled = True or (image, id) pair if labeld = False

    return dataset



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    return image, label   



def get_training_dataset(dataset):

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset(dataset):

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES) * 0.75

# use validation data for training

NUM_VALIDATION_IMAGES = count_data_items(TRAINING_FILENAMES) * 0.25

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE



print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
from tensorflow.keras.applications import DenseNet201



def get_model():

    

    with strategy.scope():

        rnet = DenseNet201(

            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),

            weights='imagenet',

            include_top=False

        )



        # trainable rnet

        rnet.trainable = True



        model = tf.keras.Sequential([

            rnet,

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(len(CLASSES), activation='softmax')

        ])



    model.compile(

        optimizer='adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy']

    )



    return model



def train_cross_validate(folds = 4):

    histories = []

    models = []

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 4)

    kfold = KFold(folds, shuffle = True, random_state = 42)

    for fold, (trn_ind, val_ind) in enumerate(kfold.split(TRAINING_FILENAMES)):

        train_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[trn_ind]['TRAINING_FILENAMES']), labeled = True)

        val_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[val_ind]['TRAINING_FILENAMES']), labeled = True, ordered = True)

        checkpoint_name = f'model_fold_{fold + 1}' + '.h5'

        model_checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_name, save_best_only = True, save_weights_only = True)

        model = get_model()

        history = model.fit(

        get_training_dataset(train_dataset), 

        steps_per_epoch = STEPS_PER_EPOCH,

        epochs = EPOCHS,

        callbacks = [lr_callback, early_stopping, model_checkpoint],

        validation_data = get_validation_dataset(val_dataset)

        )

        print('Load best weights for model prediction')

        model.load_weights(checkpoint_name)

        models.append(model)

        histories.append(history)

        

    return histories, models



def train_and_predict(folds = 4):

    test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.

    test_images_ds = test_ds.map(lambda image, idnum: image)

    print('Start training {} models'.format(folds))

    histories, models = train_cross_validate(folds = folds)

    print('Computing predictions...')

    # get the mean probability of the folds models

    probabilities = np.average([models[i].predict(test_images_ds) for i in range(folds)], axis = 0)

    predictions = np.argmax(probabilities, axis=-1)

    print('Generating submission.csv file...')

    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

    np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')

    return histories, models

    

# run train and predict

histories, models = train_and_predict(folds = 4)