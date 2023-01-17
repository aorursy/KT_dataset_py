# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from matplotlib.image import imread

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import shutil
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = '/kaggle/input/humpback-whale-identification/'
import tensorflow as tf
tf.test.gpu_device_name()
from keras.models import Sequential
from keras.applications import VGG16
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
IMAGE_SIZE = (224, 224)
VALIDATION_SPLIT = 0.7
BATCH_SIZE = 32
NUM_CLASSES = 5005
EPOCHS = 10
# load the class label and image name file
label_file = pd.read_csv(path+'/train.csv').rename(columns={'Id': 'label', 'Image': 'filename'})
# create Image generator for data augmentation
image_data_gen = ImageDataGenerator(
    rescale=1./255.,
    width_shift_range=[+0.2, 0, -0.2],
    height_shift_range=[+0.2, 0, -0.2],
    rotation_range=30,
    fill_mode="nearest",
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=VALIDATION_SPLIT
)
# read the data directly from the directory
train_gen = image_data_gen.flow_from_dataframe(
    dataframe=label_file,
    directory=path+'/train',
    x_col="filename",
    y_col="label",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode="categorical",
    batch_size=BATCH_SIZE,
    shuffle=True,
    interpolation="nearest",
    validate_filenames=True
)
# create the model
vgg16_model = Sequential()

# add the VGG16 layers with weights
vgg16_model.add(VGG16(
    include_top=False,
    weights="imagenet",
    classes=NUM_CLASSES
))

vgg16_model.add(GlobalAveragePooling2D())

# add the dense layer
vgg16_model.add(Dense(units=NUM_CLASSES, activation='softmax'))


vgg16_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

vgg16_model.summary()
history = vgg16_model.fit_generator(generator=train_gen, steps_per_epoch=train_gen.n//EPOCHS, epochs=EPOCHS)