import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os # accessing directory structure
import sys
import random
import tensorflow as tf
import keras_preprocessing
from tensorflow.keras.preprocessing import image
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.layers.normalization import BatchNormalization
tf.__version__
# Any results you write to the current directory are saved as output.
TRAINING_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/train/' 
VALIDATION_DIR = '/kaggle/input/tomato/New Plant Diseases Dataset(Augmented)/valid/'

# this is the augmentation configuration we will use for training
train_gen = ImageDataGenerator(
rescale = 1./255,
rotation_range=40,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True,
fill_mode='nearest')

valid_gen = ImageDataGenerator(rescale = 1./255)
 
#rgb --> the images will be converted to have 3 channels.

train_data = train_gen.flow_from_directory(
TRAINING_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb",
batch_size=128
)

valid_data = valid_gen.flow_from_directory(
VALIDATION_DIR,
target_size=(227,227),
class_mode='categorical',
color_mode="rgb"
)
#
# Initializing the CNN based AlexNet
model = Sequential()

#valid:zero padding, same:keep same dimensionality by add padding

# Convolution Step 1
model.add(Convolution2D(96, 11, strides = (4, 4), padding = 'valid', input_shape=(227, 227, 3), activation = 'relu'))

# Max Pooling Step 1
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))
model.add(BatchNormalization())

# Convolution Step 2
model.add(Convolution2D(256, 5, strides = (1, 1), padding='same', activation = 'relu'))

# Max Pooling Step 2
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding='valid'))
model.add(BatchNormalization())

# Convolution Step 3
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))
model.add(BatchNormalization())

# Convolution Step 4
model.add(Convolution2D(384, 3, strides = (1, 1), padding='same', activation = 'relu'))
model.add(BatchNormalization())

# Convolution Step 5
model.add(Convolution2D(256, 3, strides=(1,1), padding='same', activation = 'relu'))

# Max Pooling Step 3
model.add(MaxPooling2D(pool_size = (3, 3), strides = (2, 2), padding = 'valid'))
model.add(BatchNormalization())

# Flattening Step --> 6*6*256 = 9216
model.add(Flatten())

# Full Connection Steps
# 1st Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# 2nd Fully Connected Layer
model.add(Dense(units = 4096, activation = 'relu'))
model.add(Dropout(0.4))
model.add(BatchNormalization())

# 3rd Fully Connected Layer
model.add(Dense(units = 10, activation = 'softmax'))

model.summary()
from keras.optimizers import Adam
import keras
opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
#train_num/128 = 144
#valid_num//128 = 35
train_num = train_data.n
valid_num = valid_data.n

train_batch_size = train_data.batch_size # choose 128
valid_batch_size = valid_data.batch_size #default 32

STEP_SIZE_TRAIN = train_num//train_batch_size #144
STEP_SIZE_VALID = valid_num//valid_batch_size #144

history = model.fit_generator(generator=train_data,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_data,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25
)



