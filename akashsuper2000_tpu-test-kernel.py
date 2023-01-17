# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from kaggle_datasets import KaggleDatasets

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization

import matplotlib.pyplot as plt

print(tf.__version__)
print(tf.keras.__version__)

# Any results you write to the current directory are saved as output.
AUTO = tf.data.experimental.AUTOTUNE
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


# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path()
EPOCHS = 20
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 512
ytrain = np.concatenate([np.zeros(75000),np.ones(225000)])
ytrain.shape
train_paths = ['/kaggle/input/alaska2-image-steganalysis/Cover/'+i for i in os.listdir('/kaggle/input/alaska2-image-steganalysis/Cover')]+['/kaggle/input/alaska2-image-steganalysis/UERD/'+i for i in os.listdir('/kaggle/input/alaska2-image-steganalysis/UERD')]+['/kaggle/input/alaska2-image-steganalysis/JUNIWARD/'+i for i in os.listdir('/kaggle/input/alaska2-image-steganalysis/JUNIWARD')]+['/kaggle/input/alaska2-image-steganalysis/JMiPOD/'+i for i in os.listdir('/kaggle/input/alaska2-image-steganalysis/JMiPOD')]
len(train_paths)
test_paths = ['/kaggle/input/alaska2-image-steganalysis/Test/'+i for i in os.listdir('/kaggle/input/alaska2-image-steganalysis/Test')]
len(test_paths)
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, ytrain))
    .map(decode_image, num_parallel_calls=AUTO)
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)
LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def get_model():
    model = Sequential()

    model.add(Conv2D(20,kernel_size=(5,5),activation='relu',input_shape=(512,512,3),padding='same'))
    model.add(Dropout(0.2))
    model.add(MaxPool2D((2,2),strides=(2,2)))


    model.add(Conv2D(50,kernel_size=(5,5),activation='relu',padding='same'))
    model.add(Dropout(0.2))

    model.add(MaxPool2D((2,2),strides=(2,2)))



    model.add(Flatten())

    model.add(Dense(512,activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(7,activation='softmax'))
    
    return model

with strategy.scope():
    model = get_model()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
model.fit(
    train_dataset, 
    steps_per_epoch=ytrain.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=EPOCHS
)
probs = model.predict(test_dataset)
