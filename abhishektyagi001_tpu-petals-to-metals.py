# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Detect hardware, return appropriate distribution strategy

import tensorflow as tf

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



print("REPLICAS: ", strategy.num_replicas_in_sync)
import tensorflow as tf

from kaggle_datasets import KaggleDatasets

import numpy as np



print("Tensorflow version " + tf.__version__)



GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
import glob

import warnings



import cv2



import keras

import keras.backend as K



from keras import Model, Sequential

from keras.layers import Dense, Dropout, Flatten, Input, LeakyReLU

from keras.layers import BatchNormalization, Activation, Conv2D

from keras.layers import GlobalAveragePooling2D, Lambda

from keras.optimizers import Adam, RMSprop



from keras.applications.xception import Xception

from keras.applications.xception import preprocess_input

from keras.preprocessing.image import ImageDataGenerator 

from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint



from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd











ds_train = '../input/tpu-getting-started/tfrecords-jpeg-512x512/train'

ds_valid = '../input/tpu-getting-started/tfrecords-jpeg-512x512/val'

ds_test = '../input/tpu-getting-started/tfrecords-jpeg-512x512/test'



print("Training:", ds_train)

print ("Validation:", ds_valid)

print("Test:", ds_test)
from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input



with strategy.scope():

    pretrained_model = tf.keras.applications.VGG16(

        weights='imagenet',

        include_top=False ,

        input_shape=[512, 512, 3]

    )

    pretrained_model.trainable = False

    

    model = tf.keras.Sequential([

        # To a base pretrained on ImageNet to extract features from images...

        pretrained_model,

        # ... attach a new head to act as a classifier.

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(len('../input/tpu-getting-started/tfrecords-jpeg-512x512/train'), activation='softmax')

    ])

    model.compile(

        optimizer='adam',

        loss = 'sparse_categorical_crossentropy',

        metrics=['sparse_categorical_accuracy'],

    )



model.summary()
IMAGE_SIZE = [512, 512] # at this size, a GPU will run out of memory. Use the TPU

EPOCHS = 5

BATCH_SIZE = 16 * strategy.num_replicas_in_sync



NUM_TRAINING_IMAGES = 12753

NUM_TEST_IMAGES = 7382

STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

AUTO = tf.data.experimental.AUTOTUNE
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [512, 512, 3]) # explicit size needed for TPU

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



def get_training_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/train/*.tfrec'), labeled=True)

    dataset = dataset.repeat() # the training dataset must repeat for several epochs

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_validation_dataset():

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/val/*.tfrec'), labeled=True, ordered=False)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.cache()

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



def get_test_dataset(ordered=False):

    dataset = load_dataset(tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords-jpeg-512x512/test/*.tfrec'), labeled=False, ordered=ordered)

    dataset = dataset.batch(BATCH_SIZE)

    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)

    return dataset



training_dataset = get_training_dataset()

validation_dataset = get_validation_dataset()
with strategy.scope():    

    pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=[*IMAGE_SIZE, 3])

    pretrained_model.trainable = False # tramsfer learning

    

    model = tf.keras.Sequential([

        pretrained_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(104, activation='softmax')

    ])

        

model.compile(

    optimizer='adam',

    loss = 'sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)



historical = model.fit(training_dataset, 

          steps_per_epoch=STEPS_PER_EPOCH, 

          epochs=EPOCHS, 

          validation_data=validation_dataset)
test_ds = get_test_dataset(ordered=True) # since we are splitting the dataset and iterating separately on images and ids, order matters.



print('Computing predictions...')

test_images_ds = test_ds.map(lambda image, idnum: image)

probabilities = model.predict(test_images_ds)

predictions = np.argmax(probabilities, axis=-1)

print(predictions)



print('Generating submission.csv file...')

test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()

test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')