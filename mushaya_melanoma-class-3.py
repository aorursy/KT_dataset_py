#loading modules

import math, random, os, re, time

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras import layers, Model

import matplotlib.pylab as plt

from sklearn.model_selection import train_test_split, KFold

import PIL

import cv2

import seaborn as sns

from kaggle_datasets import KaggleDatasets

from tqdm import tqdm
try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.





AUTO     = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

#REPLICAS = 8

print(f'REPLICAS: {REPLICAS}')
#loading data

dirname='../input/siim-isic-melanoma-classification/'

train = pd.read_csv(dirname+'train.csv')

test = pd.read_csv(dirname + 'test.csv')

print(train.head())

print(len(train))

print(len(test))

print(train['target'].value_counts())
sns.countplot(train['target'])
GCS_PATH = KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images')

train_set = tf.io.gfile.glob(GCS_PATH + '/train*.tfrec')

test_filenames = tf.io.gfile.glob(GCS_PATH + '/test*.tfrec')
train_filenames , valid_filenames = train_test_split(train_set , test_size=0.2,shuffle=True)
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

IMAGE_SIZE = [512,512]

AUTO = tf.data.experimental.AUTOTUNE

imSize = 512
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

    image = tf.image.random_hue(image,0.15)

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
with strategy.scope():

    model = tf.keras.Sequential()

    model.add(tf.keras.applications.Xception(include_top=False, weights='imagenet',

                                           input_tensor=layers.Input((512, 512, 3)),

                                           input_shape=(512,512,3), pooling='avg'))

    model.add(layers.Flatten())

    model.add(layers.Dense(4096, activation='relu', name='fc1'))

    model.add(layers.Dense(1024, activation='relu', name='fc2'))

    model.add(layers.Dense(1, activation='sigmoid', name='answer'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights('../input/melanoma-tfk-rs50/melanoma_wg.h5')
model.summary()
tf.keras.utils.plot_model(

    model, to_file='model.png', show_shapes=False, show_layer_names=True,

    rankdir='TB', expand_nested=False, dpi=96)
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

def callback():

    cb = []

    

    checkpoint = ModelCheckpoint('/kaggle/working'+'/melanoma_wg.h5',

                                 save_best_only=True,

                                 mode='min',

                                 monitor='val_loss',

                                 save_weights_only=True, verbose=1)

    cb.append(checkpoint)

    

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',

                                   factor=0.3, patience=3,

                                   verbose=1, mode='auto',

                                   epsilon=0.0001, cooldown=1, min_lr=0.000001)

    cb.append(reduceLROnPlat)

    return cb
cb = callback()

histories = []

folds = 4

#train and validate

epochs = 7

for i in range(folds):

    print("Fold ", i+1)

    train_filenames , valid_filenames = train_test_split(train_set , test_size=0.2,shuffle=True)

    history = model.fit(get_training_dataset(), 

                        epochs=epochs, verbose=True, 

                        steps_per_epoch=NUM_TRAINING_IMAGES // BATCH_SIZE,

                        validation_data = get_val_dataset(), 

                        validation_steps =NUM_TEST_IMAGES//BATCH_SIZE, 

                        callbacks=cb)

    histories.append(history)
columns = 1

rows = folds

fig = plt.figure(figsize = (15,10))

i=1

for history in histories:

    graph = fig.add_subplot(rows, columns, i)

    graph.plot(history.history['accuracy'])

    graph.plot(history.history['val_accuracy'])

    graph.set_title('model accuracy')

    graph.set_ylabel('accuracy')

    graph.set_xlabel('epoch')

    graph.legend(['train', 'test'], loc='upper left')

    i+=1

plt.show()
fig = plt.figure(figsize = (15,10))

i=1

for history in histories:

    graph = fig.add_subplot(rows, columns, i)

    graph.plot(history.history['loss'])

    graph.plot(history.history['val_loss'])

    graph.set_title('model loss')

    graph.set_ylabel('loss')

    graph.set_xlabel('epoch')

    graph.legend(['train', 'test'], loc='upper left')

    i+=1

plt.show()
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

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, probabilities]), fmt=['%s', '%f'], delimiter=',', header='image_name,target', comments='')