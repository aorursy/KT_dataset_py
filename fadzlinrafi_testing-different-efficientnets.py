DEVICE = "TPU"

BASEPATH = "../input/siim-isic-melanoma-classification"
!pip install -q efficientnet
import numpy as np

import pandas as pd

import os

import random, re, math, time

random.seed(a=42)



from os.path import join 



import tensorflow as tf

import tensorflow.keras.backend as K

#import tensorflow_addons as tfa

import efficientnet.tfkeras as efn



from tqdm.keras import TqdmCallback



from PIL import Image

import PIL



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold



from sklearn.utils.class_weight import compute_class_weight



import plotly

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



from pandas_summary import DataFrameSummary



from kaggle_datasets import KaggleDatasets



from tqdm import tqdm
if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    

print("REPLICAS: ", strategy.num_replicas_in_sync)

AUTO = tf.data.experimental.AUTOTUNE

# Configuration

EPOCHS = 6

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [384, 384]
df_train = pd.read_csv(os.path.join(BASEPATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(BASEPATH, 'test.csv'))

sub = pd.read_csv(os.path.join(BASEPATH, 'sample_submission.csv'))



GCS_PATH = KaggleDatasets().get_gcs_path('melanoma-384x384')

TRAINING_FILENAMES = np.array(tf.io.gfile.glob(GCS_PATH + '/train*.tfrec'))

TEST_FILENAMES = np.array(tf.io.gfile.glob(GCS_PATH + '/test*.tfrec'))



CLASSES = [0,1]   
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



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



def data_augment(image, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),

    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    image = tf.image.random_hue(image, 0.01)

    image = tf.image.random_saturation(image, 0.7, 1.3)

    image = tf.image.random_contrast(image, 0.8, 1.2)

    image = tf.image.random_brightness(image, 0.1)

    return image, label   



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



def get_test_dataset(ordered=False):

    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))
def lrfn(epoch):

    LR_START          = 0.000005

    LR_MAX            = 0.000020 * strategy.num_replicas_in_sync

    LR_MIN            = 0.000001

    LR_RAMPUP_EPOCHS = 5

    LR_SUSTAIN_EPOCHS = 0

    LR_EXP_DECAY = .8

    

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout

from tensorflow.keras.regularizers import l2

#reg_l2 = 0.001
# Efficeint Net B0



with strategy.scope():

    model0 = tf.keras.Sequential([

        efn.EfficientNetB0(

            input_shape=(*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(1, activation='sigmoid')

    ])

    

model0.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)



# Efficeint Net B1



with strategy.scope():

    model1 = tf.keras.Sequential([

        efn.EfficientNetB1(

            input_shape=(*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(1, activation='sigmoid')

    ])

    

model1.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)

# Efficeint Net B2



with strategy.scope():

    model2 = tf.keras.Sequential([

        efn.EfficientNetB2(

            input_shape=(*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(1, activation='sigmoid')

    ])

    

model2.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)

# Efficeint Net B3



with strategy.scope():

    model3 = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(1, activation='sigmoid')

    ])

    

model3.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)

# Efficeint Net B4



with strategy.scope():

    model4 = tf.keras.Sequential([

        efn.EfficientNetB4(

            input_shape=(*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(1, activation='sigmoid')

    ])

    

model4.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)



# Efficeint Net B6



with strategy.scope():

    model6 = tf.keras.Sequential([

        efn.EfficientNetB6(

            input_shape=(*IMAGE_SIZE, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(1, activation='sigmoid')

    ])

    

model6.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)



history0 = model0.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule])

history1 = model1.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule])

history2 = model2.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule])

history3 = model3.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule])

history4 = model4.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule])

history6 = model6.fit(get_training_dataset(), steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule])
test_ds = get_test_dataset(ordered=True)



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)



probabilities0 = model0.predict(test_images_ds)

probabilities1 = model1.predict(test_images_ds)

probabilities2 = model2.predict(test_images_ds)

probabilities3 = model3.predict(test_images_ds)

probabilities4 = model4.predict(test_images_ds)

probabilities6 = model6.predict(test_images_ds)
print('Generating submission.csv files...')

print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()



test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
pred_df0 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities0)})

pred_df1 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities1)})

pred_df2 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities2)})

pred_df3 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities3)})

pred_df4 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities4)})

pred_df6 = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities6)})



pred_df6.head()
sub0 = sub.copy()

sub1 = sub.copy()

sub2 = sub.copy()

sub3 = sub.copy()

sub4 = sub.copy()

sub6 = sub.copy()

sub4.head()
del sub0['target']

sub0 = sub0.merge(pred_df0, on='image_name')

sub0.to_csv('submission0.csv', index=False)



del sub1['target']

sub1 = sub1.merge(pred_df1, on='image_name')

sub1.to_csv('submission1.csv', index=False)



del sub2['target']

sub2 = sub2.merge(pred_df2, on='image_name')

sub2.to_csv('submission2.csv', index=False)



del sub3['target']

sub3 = sub3.merge(pred_df3, on='image_name')

sub3.to_csv('submission3.csv', index=False)



del sub4['target']

sub4 = sub4.merge(pred_df4, on='image_name')

sub4.to_csv('submission4.csv', index=False)



del sub6['target']

sub6 = sub6.merge(pred_df6, on='image_name')

sub6.to_csv('submission6.csv', index=False)



sub4.head()
#average



ensemble1 = (sub0['target'] + sub1['target'] + sub2['target'] + sub3['target'] + sub4['target'] + sub6['target'])/6

ensemble_img1 = sub1['image_name']

ensemble_sub1 = pd.concat([ensemble_img1, ensemble1], axis = 1)

ensemble_sub1.to_csv('submission7.csv', index=False)

ensemble_sub1.head()