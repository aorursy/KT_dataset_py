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

EPOCHS = 8

BATCH_SIZE = 8 * strategy.num_replicas_in_sync



IMAGE_SIZE1 = [256, 256]

IMAGE_SIZE2 = [384, 384]

IMAGE_SIZE3 = [512, 512]

IMAGE_SIZE4 = [768, 768]

df_train = pd.read_csv(os.path.join(BASEPATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(BASEPATH, 'test.csv'))

sub = pd.read_csv(os.path.join(BASEPATH, 'sample_submission.csv'))



CLASSES = [0,1]   
GCS_PATH1 = KaggleDatasets().get_gcs_path('melanoma-256x256')

TRAINING_FILENAMES1 = np.array(tf.io.gfile.glob(GCS_PATH1 + '/train*.tfrec'))

TEST_FILENAMES1 = np.array(tf.io.gfile.glob(GCS_PATH1 + '/test*.tfrec'))



GCS_PATH2 = KaggleDatasets().get_gcs_path('melanoma-384x384')

TRAINING_FILENAMES2 = np.array(tf.io.gfile.glob(GCS_PATH2 + '/train*.tfrec'))

TEST_FILENAMES2 = np.array(tf.io.gfile.glob(GCS_PATH2 + '/test*.tfrec'))



GCS_PATH3 = KaggleDatasets().get_gcs_path('melanoma-512x512')

TRAINING_FILENAMES3 = np.array(tf.io.gfile.glob(GCS_PATH3 + '/train*.tfrec'))

TEST_FILENAMES3 = np.array(tf.io.gfile.glob(GCS_PATH3 + '/test*.tfrec'))



GCS_PATH4 = KaggleDatasets().get_gcs_path('melanoma-768x768')

TRAINING_FILENAMES4 = np.array(tf.io.gfile.glob(GCS_PATH4 + '/train*.tfrec'))

TEST_FILENAMES4 = np.array(tf.io.gfile.glob(GCS_PATH4 + '/test*.tfrec'))
import cv2
def decode_image1(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE1, 3])

    return image



def read_labeled_tfrecord1(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image1(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord1(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image1(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



def load_dataset1(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord1 if labeled else read_unlabeled_tfrecord1, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_training_dataset1():

    dataset = load_dataset1(TRAINING_FILENAMES1, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset1(ordered=False):

    dataset = load_dataset1(TEST_FILENAMES1, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
def decode_image2(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE2, 3])

    return image



def read_labeled_tfrecord2(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image2(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord2(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image2(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



def load_dataset2(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord2 if labeled else read_unlabeled_tfrecord2, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_training_dataset2():

    dataset = load_dataset2(TRAINING_FILENAMES2, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset2(ordered=False):

    dataset = load_dataset2(TEST_FILENAMES2, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
def decode_image3(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE3, 3])

    return image



def read_labeled_tfrecord3(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image3(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord3(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image3(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



def load_dataset3(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord3 if labeled else read_unlabeled_tfrecord3, num_parallel_calls=AUTO)

    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False

    return dataset



def get_training_dataset3():

    dataset = load_dataset3(TRAINING_FILENAMES3, labeled=True)

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset3(ordered=False):

    dataset = load_dataset3(TEST_FILENAMES3, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset
def decode_image4(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [*IMAGE_SIZE4, 3])

    return image



def read_labeled_tfrecord4(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        #"class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image4(example['image'])

    #label = tf.cast(example['class'], tf.int32)

    label = tf.cast(example['target'], tf.int32)

    return image, label # returns a dataset of (image, label) pairs



def read_unlabeled_tfrecord4(example):

    UNLABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element

        # class is missing, this competitions's challenge is to predict flower classes for the test dataset

    }

    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    image = decode_image4(example['image'])

    idnum = example['image_name']

    return image, idnum # returns a dataset of image(s)



def load_dataset4(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files at once and

    # disregarding data order. Order does not matter since we will be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        ignore_order.experimental_deterministic = False # disable order, increase speed



    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files

    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.map(read_labeled_tfrecord4 if labeled else read_unlabeled_tfrecord4, num_parallel_calls=AUTO)

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



def get_training_dataset4():

    dataset = load_dataset4(TRAINING_FILENAMES4, labeled=True)

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



def get_test_dataset4(ordered=False):

    dataset = load_dataset4(TEST_FILENAMES4, labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def count_data_items(filenames):

    # the number of data items is written in the name of the .tfrec files, i.e. flowers00-230.tfrec = 230 data items

    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]

    return np.sum(n)



NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES1)

NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES1)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))
training_set1 = get_training_dataset1()

training_set2 = get_training_dataset2()

training_set3 = get_training_dataset3()

training_set4 = get_training_dataset4()
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

reg_l2 = 0.001
#L2 tried = 0.0001/0.001/0.01

#LR = 0.0003

#opt = tf.keras.optimizers.Adam(learning_rate = LR)
"""def get_model(base_model):

        model = tf.keras.Sequential([

            base_model,

            GlobalAveragePooling2D(),

            Dense(512, activation='relu', kernel_regularizer=l2(reg_l2),

            bias_regularizer=l2(reg_l2)),

            BatchNormalization(),

            Dropout(0.2),

            Dense(182, activation='relu', kernel_regularizer=l2(reg_l2),

            bias_regularizer=l2(reg_l2)),

            BatchNormalization(),

            Dropout(0.2),

            Dense(1, activation='sigmoid')

        ])

        return model



def compile_model(base_model):

        with strategy.scope():

            model = get_model(base_model)

            

            model.compile(optimizer='adam',loss = 'binary_crossentropy',metrics=['accuracy'])

        return model

    

model1 = compile_model(base_model1)

model2 = compile_model(base_model2)

model3 = compile_model(base_model3)

model4 = compile_model(base_model4)"""
with strategy.scope():

    model1 = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(*IMAGE_SIZE1, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(182, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(1, activation='sigmoid')

    ])

    

model1.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)

with strategy.scope():

    model2 = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(*IMAGE_SIZE2, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(182, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(1, activation='sigmoid')

    ])

    

model2.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)
with strategy.scope():

    model3 = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(*IMAGE_SIZE3, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(182, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(1, activation='sigmoid')

    ])

    

model3.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)
with strategy.scope():

    model4 = tf.keras.Sequential([

        efn.EfficientNetB3(

            input_shape=(*IMAGE_SIZE4, 3),

            weights='imagenet',

            include_top=False

        ),

        GlobalAveragePooling2D(),

        Dense(512, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(182, activation='relu', kernel_regularizer=l2(reg_l2),

    bias_regularizer=l2(reg_l2)),

        BatchNormalization(),

        Dropout(0.2),

        Dense(1, activation='sigmoid')

    ])

    

model4.compile(

    optimizer='adam',

    loss = 'binary_crossentropy',

    metrics=['accuracy']

)
lr_schedule1 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

lr_schedule2 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

lr_schedule3 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

lr_schedule4 = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)



history1 = model1.fit(training_set1, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule1])

history2 = model2.fit(training_set2, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule2])

history3 = model3.fit(training_set3, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule3])

history4 = model4.fit(training_set4, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, callbacks=[lr_schedule4])
test_ds1 = get_test_dataset1(ordered=True)

test_ds2 = get_test_dataset2(ordered=True)

test_ds3 = get_test_dataset3(ordered=True)

test_ds4 = get_test_dataset4(ordered=True)



print('Computing predictions...')

test_images_ds1 = test_ds1.map(lambda image, idnum: image)

test_images_ds2 = test_ds2.map(lambda image, idnum: image)

test_images_ds3 = test_ds3.map(lambda image, idnum: image)

test_images_ds4 = test_ds4.map(lambda image, idnum: image)



probabilities1 = model1.predict(test_images_ds1)

probabilities2 = model2.predict(test_images_ds2)

probabilities3 = model3.predict(test_images_ds3)

probabilities4 = model4.predict(test_images_ds4)
print('Generating submission.csv files...')

test_ids_ds1 = test_ds1.map(lambda image, idnum: idnum).unbatch()

test_ids_ds2 = test_ds2.map(lambda image, idnum: idnum).unbatch()

test_ids_ds3 = test_ds3.map(lambda image, idnum: idnum).unbatch()

test_ids_ds4 = test_ds4.map(lambda image, idnum: idnum).unbatch()



test_ids1 = next(iter(test_ids_ds1.batch(NUM_TEST_IMAGES))).numpy().astype('U')

test_ids2 = next(iter(test_ids_ds2.batch(NUM_TEST_IMAGES))).numpy().astype('U')

test_ids3 = next(iter(test_ids_ds3.batch(NUM_TEST_IMAGES))).numpy().astype('U')

test_ids4 = next(iter(test_ids_ds4.batch(NUM_TEST_IMAGES))).numpy().astype('U')# all in one batch
pred_df1 = pd.DataFrame({'image_name': test_ids1, 'target': np.concatenate(probabilities1)})

pred_df2 = pd.DataFrame({'image_name': test_ids2, 'target': np.concatenate(probabilities2)})

pred_df3 = pd.DataFrame({'image_name': test_ids3, 'target': np.concatenate(probabilities3)})

pred_df4 = pd.DataFrame({'image_name': test_ids4, 'target': np.concatenate(probabilities4)})



pred_df4.head()
sub1 = sub.copy()

sub2 = sub.copy()

sub3 = sub.copy()

sub4 = sub.copy()



sub4.head()
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

sub4.head()
#average



ensemble1 = (sub1['target'] + sub2['target'] + sub3['target'] + sub4['target'])/4

ensemble_img1 = sub1['image_name']

ensemble_sub1 = pd.concat([ensemble_img1, ensemble1], axis = 1)

ensemble_sub1.to_csv('submission5.csv', index=False)

ensemble_sub1.head()
# weighted average??



ensemble2 = 0.15 * sub1['target'] + 0.2 * sub2['target']  + 0.3 * sub3['target'] + 0.35 * sub4['target']

ensemble_img2 = sub1['image_name']

ensemble_sub2 = pd.concat([ensemble_img2, ensemble2], axis = 1)

ensemble_sub2.to_csv('submission6.csv', index=False)

ensemble_sub2.head()
#random mixup??



frac1 = sub1.sample(frac = 0.25)

frac2 = sub2.sample(frac = 0.25)

frac3 = sub3.sample(frac = 0.25)

frac4 = sub4.sample(frac = 0.25)



ensemble3 = pd.concat([frac1,frac2,frac3,frac4], axis = 0)

ensemble3_sub = sub.copy()



ensemble3_sub = ensemble3_sub.merge(ensemble3, on='image_name')



ensemble3_sub['target'] = ensemble3_sub['target_y']

ensemble3_sub = ensemble3_sub.drop(['target_x'], axis = 1)

ensemble3_sub = ensemble3_sub.drop(['target_y'], axis = 1)



ensemble3_sub.to_csv('submission7.csv', index=False)

ensemble3_sub.head()
