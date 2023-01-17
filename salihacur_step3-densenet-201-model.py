# import libraries

import json

import math

import os

from glob import glob 

from tqdm import tqdm

from PIL import Image

import cv2 # image processing

import numpy as np # linear algebra

import pandas as pd # data processing

import matplotlib.pyplot as plt # data visualization





from sklearn.metrics import accuracy_score 

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report 

from sklearn.model_selection import train_test_split



from keras import layers

from keras.models import Sequential

from keras.optimizers import Adam, RMSprop

from keras.applications import DenseNet201

from keras.utils.np_utils import to_categorical

from keras.layers import  Conv2D,MaxPooling2D,Activation,Dropout,Flatten,Dense,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator,image,img_to_array,load_img
input_dir = "../input/chest-xray-pneumonia/chest_xray/"

train_dir = input_dir +"train/"

test_dir = input_dir +"test/"

val_dir = input_dir +"val/"
fig, ax = plt.subplots(2,3, figsize=(15,7))

ax = ax.ravel()

plt.tight_layout()

for i, _set in enumerate(['train','test','val']):

    set_path = input_dir+_set

    ax[i].axis('off')

    ax[i].imshow(plt.imread(set_path+'/NORMAL/'+os.listdir(set_path+'/NORMAL')[0]))

    ax[i].set_title('File: {} - Condition: Healthy'.format(_set))

    ax[i+3].axis('off')

    ax[i+3].imshow(plt.imread(set_path+'/PNEUMONIA/'+os.listdir(set_path+'/PNEUMONIA')[0]))

    ax[i+3].set_title('File: {} - Condition: Infected'.format(_set))
# Data Augmentation

train_datagen = ImageDataGenerator(

      rescale=1./255,

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

test_generator = test_datagen.flow_from_directory(

        test_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(

        val_dir,

        target_size=(150, 150),

        batch_size=20,

        class_mode='binary')
# Create DenseNet201 Model with Keras library



def build_model(backbone, lr=1e-4):

    model = Sequential()

    model.add(backbone)

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dropout(0.5))

    model.add(layers.BatchNormalization())

    model.add(layers.Dense(1, activation='sigmoid'))

    

    model.compile(

        loss='binary_crossentropy',

        optimizer=Adam(lr=lr),

        metrics=['acc']

    )

    return model



densenet = DenseNet201(

    weights='imagenet',

    include_top=False,

    input_shape=(150,150,3)

)



model = build_model(densenet ,lr = 1e-4)

model.summary()
# Fit Model

history = model.fit_generator(

    train_generator,

    steps_per_epoch=250,

    epochs=25,

    validation_data=test_generator,

    validation_steps=10

)
# model save

model.save_weights("cnn-example-pneumonia-densenet201.h5")
# Visualize Loss and Accuracy Rates

fig, ax = plt.subplots(1, 2, figsize=(10, 3))

ax = ax.ravel()



for i, met in enumerate(['acc', 'loss']):

    ax[i].plot(history.history[met])

    ax[i].plot(history.history['val_' + met])

    ax[i].set_title('Model {}'.format(met))

    ax[i].set_xlabel('epochs')

    ax[i].set_ylabel(met)

    ax[i].legend(['train', 'val'])