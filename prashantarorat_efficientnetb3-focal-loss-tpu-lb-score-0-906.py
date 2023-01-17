import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn
import keras
from keras.layers import *
from keras.applications import *
from keras.models import *
from keras.activations import *
import random
import tensorflow as tf
import cv2
try:
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
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
GCS_PATH = KaggleDatasets().get_gcs_path()

train_filenames = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
test_filenames = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
from sklearn.model_selection import train_test_split
train_filenames , valid_filenames = train_test_split(train_filenames , test_size=0.2,shuffle=True)
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [512,512]
AUTO = tf.data.experimental.AUTOTUNE
imSize = 512
import re

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    image = tf.image.resize(image, [imSize,imSize])
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
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
    image = tf.image.random_saturation(image, 0, 2)
    # image = tf.image.random_hue(image,0.15)
    return image, label   

def get_training_dataset():
    dataset = load_dataset(train_filenames, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_val_dataset():
    dataset = load_dataset(valid_filenames, labeled=True)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(train_filenames)
NUM_TEST_IMAGES = count_data_items(valid_filenames)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} labeled validation images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))
for image, label in get_training_dataset().take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())
# print("Test data shapes:")
!pip install efficientnet
!pip install tensorflow_addons
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model , load_model
import efficientnet.tfkeras as efn
import math
import tensorflow_addons as tfa

EPOCHS = 25

def focal_loss(gamma=2., alpha=.25):
  def focal_loss_fixed(y_true, y_pred):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
  
  return focal_loss_fixed

def get_cosine_schedule_with_warmup(lr,num_warmup_steps, num_training_steps, num_cycles=0.75):
  
    def lrfn(epoch):
        if epoch < num_warmup_steps:
            return (float(epoch) / float(max(1, num_warmup_steps))) * lr

        progress = float(epoch - num_warmup_steps ) / float(max(1, num_training_steps - num_warmup_steps))

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))) * lr

    return tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

lr_schedule= get_cosine_schedule_with_warmup(lr=0.00004,num_warmup_steps=4,num_training_steps=EPOCHS)

with strategy.scope():
    
    input = Input(shape=(512,512,3))

    base_model = efn.EfficientNetB3(weights="imagenet",include_top=False,input_shape=(512,512,3))
    base_model.trainable = True
    
    output = base_model(input)
    output = GlobalMaxPooling2D()(output)
    output = Dense(256)(output)
    outptu = LeakyReLU(alpha = 0.25)(output)
    output = Dropout(0.25)(output)

    output = Dense(16,activation="relu")(output)
    output = Dropout(0.15)(output)

    output = Dense(1,activation="sigmoid")(output)
    
    model = Model(input,output)
    
    model.compile(
        optimizer='adam',
        loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO),
        metrics=[tf.keras.metrics.AUC()]
    )
    model.summary()

model.fit(get_training_dataset(),
          epochs=EPOCHS,
          verbose=True,
          steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,
          validation_data=get_val_dataset(),
          callbacks=[lr_schedule])
num_test_images = count_data_items(test_filenames)
num_test_images
def get_test_dataset(ordered=False):
    dataset = load_dataset(test_filenames, labeled=False,ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

test_dataset = get_test_dataset(ordered=True)
print('Computing predictions...')
test_images_ds = test_dataset.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds).flatten()
print(probabilities)


print('Generating submission.csv file...')
test_ids_ds = test_dataset.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(num_test_images))).numpy().astype('U') # all in one batch
np.savetxt('EfficientNetB3_focal_v1.csv', np.rec.fromarrays([test_ids, probabilities]), fmt=['%s', '%f'], delimiter=',', header='image_name,target', comments='')
