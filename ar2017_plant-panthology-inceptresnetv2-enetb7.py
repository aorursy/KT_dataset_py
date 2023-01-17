# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install -q efficientnet

import tensorflow as tf

import keras

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets

from tensorflow.keras.models import Sequential

import tensorflow.keras.layers as L

from tensorflow.keras.applications import InceptionResNetV2

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# Detect hardware, return appropriate distribution strategy

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
GCS_DS_PATH = KaggleDatasets().get_gcs_path() # you can list the bucket with "!gsutil ls $GCS_DS_PATH"
def format_path(fn):

    return GCS_DS_PATH + '/images/' + fn + '.jpg'



train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')

test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')



train_paths = train['image_id'].apply(format_path).values

test_paths = test['image_id'].apply(format_path).values



train_labels = train.loc[:, 'healthy':].values



list_train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

list_test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
AUTOTUNE = tf.data.experimental.AUTOTUNE

EPOCHS = 40

BATCH_SIZE_1 = 16 * strategy.num_replicas_in_sync

BATCH_SIZE_2 = 64

image_size = 800
def decode_image(filename, label=None, image_size=(image_size, image_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    

    if label is None:

        return image

    else:

        return image, label

    

def data_augment(image, label=None):

    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_flip_up_down(image)

    

    if label is None:

        return image

    else:

        return image, label

    

def prepare_for_training(ds, batch_size, cache=True, shuffle_buffer_size=512):

    if cache:

        if isinstance(cache, str):

            ds = ds.cache(cache)

        else:

            ds = ds.cache()

    ds = ds.map(data_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    ds = ds.repeat()

    ds = ds.batch(batch_size)

    ds = ds.prefetch(buffer_size=AUTOTUNE)

    

    return ds
train_ds1 = list_train_ds.map(decode_image, num_parallel_calls=AUTOTUNE)

train_ds1 = prepare_for_training(train_ds1, BATCH_SIZE_1)



train_ds2 = list_train_ds.map(decode_image, num_parallel_calls=AUTOTUNE)

train_ds2 = prepare_for_training(train_ds2, BATCH_SIZE_2)



test_ds = list_test_ds.map(decode_image, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE_1)
with strategy.scope():

    model1 = tf.keras.Sequential([

        InceptionResNetV2(

            input_shape=(image_size, image_size, 3),

            weights='imagenet',

            include_top=False

        ),

        L.GlobalAveragePooling2D(),

        L.Dense(train_labels.shape[1], activation='softmax')

    ])

        

    model1.compile(

        optimizer = 'adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    model1.summary()
with strategy.scope():

    model2 = tf.keras.Sequential([

        efn.EfficientNetB7(

            input_shape=(image_size, image_size, 3),

            weights='imagenet',

            include_top=False

        ),

        L.GlobalAveragePooling2D(),

        L.Dense(train_labels.shape[1], activation='softmax'),

    ])

        

    model2.compile(

        optimizer = 'adam',

        loss = 'categorical_crossentropy',

        metrics=['categorical_accuracy']

    )

    model2.summary()
LR_START = 0.0001

LR_MAX = 0.00005 * strategy.num_replicas_in_sync

LR_MIN = 0.0001

LR_RAMPUP_EPOCHS = 4

LR_SUSTAIN_EPOCHS = 6

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = np.random.random_sample() * LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE_1



history1 = model1.fit(

    train_ds1, 

    epochs=EPOCHS, 

    callbacks=[lr_callback],

    steps_per_epoch=STEPS_PER_EPOCH,

)
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE_2



history2 = model2.fit(

    train_ds2, 

    epochs=EPOCHS, 

    callbacks=[lr_callback],

    steps_per_epoch=STEPS_PER_EPOCH,

)
probs1 = model1.predict(test_ds, verbose=1)

probs2 = model2.predict(test_ds, verbose=1)

probs_avg = (probs1 + probs2) / 2

sub.loc[:, 'healthy':] = probs_avg

sub.to_csv('submission.csv', index=False)

sub.head()