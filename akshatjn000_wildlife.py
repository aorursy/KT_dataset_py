#Importing

import os

import pandas as pd

import numpy as np

import glob

import shutil

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator



if not os.path.isdir('models'):

    os.mkdir('models')

    

print('TensorFlow version:', tf.__version__)

print('Is using GPU?', tf.test.is_gpu_available())

from os import listdir

from os.path import isfile, join, isdir
df_images = pd.DataFrame(columns=['Image', 'Class'])

base_dir = '/kaggle/input/wildlifeimage-classification/'
Classes = listdir(base_dir) 

Classes
for Class in Classes:

  temp = pd.DataFrame(os.listdir(base_dir+Class), columns=['Image'])

  temp['Image'] = (base_dir+Class+'/')+temp

  temp['Class']= Class

  df_images = df_images.append(temp)
df_images.head()
df_images.tail()
len(df_images)
df_images = df_images.sample(frac=1).reset_index(drop=True)

df_images.head()
num_train = int(round(len(df_images)*0.7))

train, test = df_images[:num_train], df_images[num_train:]

train.reset_index(drop=True)

test.reset_index(drop=True)
datagen=ImageDataGenerator(

    rescale=1./255.,validation_split=0.25, rotation_range=90, 

    width_shift_range=0.5,

    height_shift_range=0.5,

    brightness_range=[.5,1],

    shear_range=0.7,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    )



train_generator=datagen.flow_from_dataframe(

    dataframe=train,

    directory=base_dir,

    x_col="Image",

    y_col="Class",

    subset="training",

    batch_size=32,

    seed=42,

    shuffle=True,

    classes= Classes,

    class_mode="categorical",

    target_size=(256,256),

)

valid_generator=datagen.flow_from_dataframe(

    dataframe=train,

    directory=base_dir,

    x_col="Image",

    y_col="Class",

    subset="validation",

    batch_size=32,

    seed=42,

    classes= Classes,

    shuffle=True,

    class_mode="categorical",

    target_size=(256,256)

)

test_datagen=ImageDataGenerator(

    rescale=1./255.,validation_split=0.25, rotation_range=90, 

    width_shift_range=0.5,

    height_shift_range=0.5,

    brightness_range=[.5,1],

    shear_range=0.7,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True,

    )



test_generator=test_datagen.flow_from_dataframe(

    dataframe=test,

    directory=base_dir,

    x_col="Image",

    y_col=None,

    batch_size=32,

    seed=42,

    shuffle=False,

    classes= None,

    class_mode=None,

    target_size=(256,256)

)
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.layers import Dropout, Flatten, Input, Dense
from keras.applications.vgg19 import VGG19

VGG19_model = VGG19(weights='imagenet', include_top=False, input_shape=(256,256,3))

model = Sequential()

model.add(VGG19_model)

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(len(Classes), activation='softmax'))

model.summary()

model.compile(

        loss='categorical_crossentropy',

        optimizer='Adadelta', metrics=['accuracy']

    )

    

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

h = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=120,

                    callbacks=[

                        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, mode='auto', verbose = 1),

    ]

)



model.save('/kaggle/working/models/model.h5')