# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import tensorflow_addons as tfa

from pathlib import Path
PROCESSOR = 'GPU'

IMAGE_SIZE = 192

NUM_TRAINING_IMAGES = 12753

path_data = Path(f'/kaggle/input/tpu-getting-started/tfrecords-jpeg-{IMAGE_SIZE}x{IMAGE_SIZE}/')

files_train = [str(path) for path in (path_data / 'train').glob('*')] 

files_validate = [str(path) for path in (path_data / 'val').glob('*')]

files_test = [str(path) for path in (path_data / 'test').glob('*')]
def decode_image(image_data):

    image = tf.io.decode_jpeg(image_data, channels=3)

    image = tf.image.convert_image_dtype(image, dtype=tf.float32, saturate=False)  # convert image to floats in [0, 1] range

    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3]) # explicit size needed for TPU

    return image



def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring

        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element

    }

    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    image = decode_image(example['image'])

    label = tf.cast(example['class'], tf.int32)

    return ({'image': image}, {'label': label}) # returns a named dataset of (image, label) pairs



batch_size = 24

train = tf.data.Dataset.from_tensor_slices(files_train)

train = (train.shuffle(16) ## shuffle files

              .interleave(tf.data.TFRecordDataset, cycle_length=4, deterministic=False,

                         num_parallel_calls=tf.data.experimental.AUTOTUNE)

              .map(read_labeled_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

              .repeat()

#               .shuffle(8) ## shuffle images

              .batch(batch_size, drop_remainder=True)

              .prefetch(tf.data.experimental.AUTOTUNE))
validate = tf.data.Dataset.from_tensor_slices(files_validate)

validate = (validate.interleave(tf.data.TFRecordDataset, cycle_length=4, deterministic=False,

                                num_parallel_calls=tf.data.experimental.AUTOTUNE)

                    .map(read_labeled_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

                    .cache()

                    .batch(8, drop_remainder=True)

                    .prefetch(tf.data.experimental.AUTOTUNE))
!pip install  efficientnet
import efficientnet.tfkeras as efn

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input, Dense, GlobalMaxPooling2D

if PROCESSOR == 'TPU':

    # Cluster Resolver for Google Cloud TPUs.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())



    # Connects to the given cluster.

    tf.config.experimental_connect_to_cluster(tpu)



    # Initialize the TPU devices.

    tf.tpu.experimental.initialize_tpu_system(tpu)



    # TPU distribution strategy implementation.

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print("REPLICAS: ", strategy.num_replicas_in_sync)

elif PROCESSOR == 'GPU':

    # GPU distribution strategy implementation.

    strategy = tf.distribute.MirroredStrategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

else:

    tfa.options.TF_ADDONS_PY_OPS = True

    # need a strategy for the next step even if it doesn't do anything.

    strategy = tf.distribute.MirroredStrategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)
with strategy.scope():

    enet = efn.EfficientNetB3(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),

                              weights='imagenet',

                              include_top=False)



    enet.trainable = False

    

    image_input = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name='image')

    x = enet(image_input)

    x = GlobalMaxPooling2D()(x)

    label_output = Dense(104, activation='softmax', name='label')(x)

    

    model = Model(inputs=[image_input], outputs=[label_output])

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

model.summary()
model.fit(train,

          validation_data=validate,

          epochs=3,

          steps_per_epoch=NUM_TRAINING_IMAGES//batch_size)
with strategy.scope():

    optimizer = tf.keras.optimizers.Adam(lr=1e-3)

    optimizer = tfa.optimizers.SWA(optimizer, start_averaging=0, average_period=20)

    

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

model.summary()
model.fit(train,

          validation_data=validate,

          epochs=3,

          steps_per_epoch=NUM_TRAINING_IMAGES//batch_size)
## replace weights with SWA weights

print("\nEvaluate before changing weights")

model.evaluate(validate, verbose=2)

optimizer.assign_average_vars(model.variables)

## one forward pass with low learning rate to adjust batch normalization

print("\nEvaluate before updating batch norm")

model.evaluate(validate, verbose=2)

optimizer = tf.keras.optimizers.SGD(lr=1e-12)

model.compile(optimizer=optimizer,

              loss='sparse_categorical_crossentropy')

model.fit(train,

          epochs=1,

          validation_data=validate,

          verbose=2)

print("\nEvaluate on the final version")

model.evaluate(validate, verbose=2)