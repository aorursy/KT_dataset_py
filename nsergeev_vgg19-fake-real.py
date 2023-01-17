# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import re

import os

import cv2

from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



import numpy as np

import pandas as pd

from keras.applications.resnet50 import ResNet50

from keras.callbacks import ModelCheckpoint

from keras.applications import VGG19

from keras import models

from keras import layers

from keras import optimizers

import tensorflow as tf

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!ls
cd /kaggle/input/hybrid-dataset-v2/val/kaggle/working/val
base_dir   = '/kaggle/input/hybrid-dataset-v2'



train_dir  = '/kaggle/input/hybrid-dataset-v2/train/kaggle/working/train'

test_dir   = '/kaggle/input/hybrid-dataset-v2/test/kaggle/working/test'

val_dir    = '/kaggle/input/hybrid-dataset-v2/val/kaggle/working/val'

data_gen = ImageDataGenerator(rescale=1./255, rotation_range=30, horizontal_flip=True)



train_gen = data_gen.flow_from_directory(

    train_224_dir,

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary',

)



test_gen = data_gen.flow_from_directory(

    test_224_dir,

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary',

)



val_gen = data_gen.flow_from_directory(

    val_224_dir,

    target_size=(224, 224),

    batch_size=64,

    class_mode='binary',

)
!ls /kaggle/input/vgg19-trained/
weights_path = '/kaggle/input/vgg19-trained/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'

conv_base = VGG19(weights=weights_path,include_top=False, input_shape=(224, 224, 3))
conv_base.summary()
for layer in conv_base.layers[:-10]:

    layer.trainable = False



for layer in conv_base.layers:

    print(layer, layer.trainable)
model = models.Sequential()

model.add(conv_base)

model.add(layers.GlobalAveragePooling2D())

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(256, activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))
model.compile(

    loss='binary_crossentropy',

    # optimizer=optimizers.Adam(lr=0.0002),

    optimizer=optimizers.RMSprop(lr=1e-4),

    metrics=['accuracy']

)
model.summary()
checkpointpath='/kaggle/working/weights.{epoch:02d}-{val_loss:.2f}.hdf5'

checkpoint = ModelCheckpoint(checkpointpath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False,mode='auto',

                            period=1)



early_stop = EarlyStopping(monitor='val_loss', 

                           min_delta=0.001, 

                           patience=10, 

                           mode='min', 

                           verbose=1)



model_history = model.fit_generator(

    train_gen,

    validation_data=val_gen,

    steps_per_epoch=train_gen.samples/train_gen.batch_size,

    validation_steps=val_gen.samples/val_gen.batch_size,

    callbacks=[checkpoint, early_stop],

    epochs=200,

    verbose=1

)
!ls -la
!ls
vgg19 = models.load_model(vgg19_weghts_path)