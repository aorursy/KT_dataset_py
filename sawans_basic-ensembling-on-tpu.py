!pip install -q efficientnet
# Installing tensorflow 2 to fix the TPU error 'Socket closed'
!pip install tensorflow==2.2-rc1
import re

import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)
# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# GCS Data access path for this dataset
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# Configuration
EPOCHS = 5
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
# Loading the datasets

sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
train.head()
#Reading filenames for TPU

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
print(len(TRAINING_FILENAMES))
VALIDATION_FILENAMES = TRAINING_FILENAMES[int(0.8*len(TRAINING_FILENAMES)):]
TRAINING_FILENAMES = TRAINING_FILENAMES[:int(0.8*len(TRAINING_FILENAMES))]
print("Number of training file names : {} and number of validation file names :{}".format(len(TRAINING_FILENAMES),len(VALIDATION_FILENAMES)))
# Function to normalize and reshape the images
def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image
# Function to read labeled training records

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
# Function to read unlabeled test records

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
# Function to load dataset

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
# Function to apply augumentation on training images

def data_augment(image, label):
    
    image = tf.image.random_flip_left_right(image)
    return image, label  
# Function to get Training dataset

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
# Function to get Validation dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
# Function to get test dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
# Getting the total numnber of training, validation and test images.
# Defining the number of steps in each epoch

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} validation images {} unlabeled test images'.format(
    NUM_TRAINING_IMAGES,NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
def build_lrfn(lr_start=0.00001, lr_max=0.000075, 
               lr_min=0.000001, lr_rampup_epochs=20, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn
# Model 1

with strategy.scope():
        model1 = tf.keras.Sequential([
            efn.EfficientNetB7(
                input_shape=(*IMAGE_SIZE, 3),
                weights='imagenet',
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
        model1.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()]
        )
        model1.summary()
# Model 2

with strategy.scope():
        model2 = tf.keras.Sequential([
            efn.EfficientNetB0(
                input_shape=(*IMAGE_SIZE, 3),
                weights='imagenet',
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
        model2.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()]
        )
        model2.summary()
from tensorflow.keras.applications import DenseNet201

# Model 3

with strategy.scope():
        dnet201 = DenseNet201(
            input_shape=(*IMAGE_SIZE, 3),
            weights='imagenet',
            include_top=False
        )
        dnet201.trainable = True

        model3 = tf.keras.Sequential([
            dnet201,
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
        model3.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=[tf.keras.metrics.AUC()]
        )
        model3.summary()
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
train_dataset = get_training_dataset()
valid_dataset = get_validation_dataset()
history1 = model1.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[lr_schedule],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset)
# create copies for each model if you want to
sub1 = sub.copy()

# Getting predictions on test data
test_ds = get_test_dataset(ordered=True)
print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model1.predict(test_images_ds)
# Generating submission file

print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
pred_df.head()
del sub1['target']
sub1 = sub1.merge(pred_df, on='image_name')
sub1.to_csv('submission_efficientnetb7.csv', index=False)
history2 = model2.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[lr_schedule],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset)

model1.save("efficientnetb0.h5")
# create copies for each model if you want to
sub2 = sub.copy()

# Getting predictions on test data
#test_ds = get_test_dataset(ordered=True)
print('Computing predictions...')
#test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model2.predict(test_images_ds)
# Generating submission file

print('Generating submission.csv file...')
#test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
#test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
pred_df.head()
del sub2['target']
sub2 = sub2.merge(pred_df, on='image_name')
sub2.to_csv('submission_efficientnetb0.csv', index=False)
history3 = model3.fit(
        train_dataset,
        epochs=EPOCHS,
        callbacks=[lr_schedule],
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=valid_dataset)
# create copies for each model if you want to
sub3 = sub.copy()

# Getting predictions on test data
#test_ds = get_test_dataset(ordered=True)
print('Computing predictions...')
#test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model3.predict(test_images_ds)
# Generating submission file

print('Generating submission.csv file...')
#test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
#test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
pred_df.head()
del sub3['target']
sub3 = sub3.merge(pred_df, on='image_name')
sub3.to_csv('submission_Densenet201.csv', index=False)
sub_efficientnetB7 = pd.read_csv('../input/second-ensemble/submission_efficientnetb7.csv')
sub_efficientnetB0 = pd.read_csv('../input/second-ensemble/submission_efficientnetb0.csv')
sub_densenet201 = pd.read_csv('../input/second-ensemble/submission_Densenet201.csv')
sub_vgg16 = pd.read_csv('../input/second-ensemble/submission_vgg16_Complete.csv')
def rank_data(sub):
    sub['target'] = sub['target'].rank() / sub['target'].rank().max()
    return sub
sub_efficientnetB7 = rank_data(sub_efficientnetB7)
sub_efficientnetB0 = rank_data(sub_efficientnetB0)
sub_densenet201 = rank_data(sub_densenet201)
sub_vgg16 = rank_data(sub_vgg16)
sub_efficientnetB7.columns = ['image_name', 'target1']
sub_efficientnetB0.columns = ['image_name', 'target2']
sub_densenet201.columns = ['image_name', 'target3']
sub_vgg16.columns = ['image_name', 'target4']
f_sub = sub_efficientnetB7.merge(sub_efficientnetB0, on = 'image_name').merge(sub_densenet201, on = 'image_name').merge(sub_vgg16, on = 'image_name')

f_sub['target'] = f_sub['target1'] * 0.4 + f_sub['target2'] * 0.4 + f_sub['target3'] * 0.02 + f_sub['target4'] * 0.02
f_sub = f_sub[['image_name', 'target']]
f_sub.to_csv('blend_sub_2.csv', index = False)
