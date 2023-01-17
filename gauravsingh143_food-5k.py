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
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.applications.vgg16 import VGG16 , preprocess_input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt
import sys, os

from glob import glob
!mkdir /kaggle/working/data
!mkdir /kaggle/working/data/train
!mkdir /kaggle/working/data/test
!mkdir /kaggle/working/data/train/food
!mkdir /kaggle/working/data/train/nonfood
!mkdir /kaggle/working/data/test/food
!mkdir /kaggle/working/data/test/nonfood
!cp /kaggle/input/food5k/Food-5K/training/0*.jpg /kaggle/working/data/train/nonfood
!cp /kaggle/input/food5k/Food-5K/training/1*.jpg /kaggle/working/data/train/food
!cp /kaggle/input/food5k/Food-5K/validation/0*.jpg /kaggle/working/data/test/nonfood
!cp /kaggle/input/food5k/Food-5K/validation/1*.jpg /kaggle/working/data/test/food
train_path = '/kaggle/working/data/train'
valid_path = '/kaggle/working/data/test'
IMAGE_SIZE = [200,200]
# number of files
train_image_files = glob(train_path + '/*/*.jpg')
valid_image_files = glob(valid_path + '/*/*.jpg')
# number of classes
folders = glob(train_path + '/*')
folders
# random image
plt.imshow(image.load_img(np.random.choice(train_image_files)))
ptm = VGG16(
    input_shape = IMAGE_SIZE + [3],
    weights = 'imagenet',
    include_top = False
)
# freezing VGG16 Model weights
ptm.trainable = False
# map data into feature vectors

K = len(folders) # no. of classes
x = Flatten()(ptm.output)
x = Dense(K, activation='softmax')(x)
model = Model(inputs=ptm.input, outputs=x)
# ImageDataGenerator
gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)
# Generator

batch_size = 128

train_generator = gen.flow_from_directory(
    train_path,
    shuffle=True,
    target_size=IMAGE_SIZE,
    batch_size=batch_size
)

valid_generator = gen.flow_from_directory(
    valid_path,
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=batch_size
)
model.compile(
    loss='categorical_crossentropy',  # as generator yields one-hit encoded results
    optimizer='adam',
    metrics=['accuracy']
)
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=10,
    steps_per_epoch=int(np.ceil(len(train_image_files)/batch_size)),
    validation_steps=int(np.ceil(len(valid_image_files)/batch_size))
)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.legend()
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='valid accuracy')
plt.legend()


