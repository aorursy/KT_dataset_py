# do the necessary imports



import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import Adam

from keras.preprocessing.image import img_to_array

from keras.utils import to_categorical



from sklearn.model_selection import train_test_split

import numpy as np

import pandas as pd

import random

import cv2

import os

from PIL import Image



from keras.layers import *

from keras.models import *

import keras

# Hyper - parameters



epochs = 100

lr = 1e-3

batch_size = 64

#img_dims = (96,96,3)



data = []

labels = []
size = 224
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

from tensorflow.keras.applications import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.image import load_img, img_to_array



resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# CALLBACKS



from keras.callbacks.callbacks import EarlyStopping

from keras.callbacks.callbacks import ReduceLROnPlateau



es = EarlyStopping(patience=5, monitor = 'val_accuracy')

rlp = ReduceLROnPlateau(patience=5, monitor = 'val_accuracy')



callbacks = [es, rlp]
train_datagen = ImageDataGenerator(horizontal_flip=True,

                                   width_shift_range = 0.4,

                                   height_shift_range = 0.4,

                                   zoom_range=0.3,

                                   rotation_range=20,

                                   rescale = 1./255

                                   )



test_gen = ImageDataGenerator(rescale = 1./255)



image_size = 224

batch_size = 64



train_generator = train_datagen.flow_from_directory(

        '../input/gender-classification-dataset/Training',

        target_size=(image_size, image_size),

        batch_size=batch_size,

        class_mode='binary')



validation_generator = test_gen.flow_from_directory(

    '../input/gender-classification-dataset/Validation',

    target_size = (image_size, image_size),

    batch_size = batch_size,

    class_mode = 'binary'

)



num_classes = len(train_generator.class_indices)

print('Numer of classes:' ,num_classes)

print('Class labels: ', train_generator.class_indices)









"""

train_datagen = ImageDataGenerator(

    rescale = 1./255, 

    shear_range = 0.2, 

    zoom_range = 0.2,

    horizontal_flip = True

)



test_datagen = ImageDataGenerator(rescale = 1./255)



train_generator = train_datagen.flow_from_directory(

    train_path, 

    target_size = (size, size),

    batch_size = batch_size,

    class_mode = 'binary'

)



validation_generator = test_datagen.flow_from_directory(

    valid_path,

    target_size = (size, size),

    batch_size = batch_size, 

    class_mode = 'binary'

)



"""
model = Sequential()



model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

model.add(Flatten())

model.add(BatchNormalization())

model.add(Dense(2048, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(1024, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))



model.layers[0].trainable = False
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit_generator(train_generator, steps_per_epoch = int(47000/64) + 1  , epochs = 30, validation_data = validation_generator, callbacks = callbacks)
model.save('model3.h5')
model2 = Sequential()



model2.add(ResNet50(include_top=False, pooling='avg', weights=None))

model2.add(Flatten())

model2.add(BatchNormalization())

model2.add(Dense(2048, activation='relu'))

model2.add(BatchNormalization())

model2.add(Dense(1024, activation='relu'))

model2.add(BatchNormalization())

model2.add(Dense(1, activation='sigmoid'))



model2.layers[0].trainable = True
model2.summary()
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model2.fit_generator(train_generator, steps_per_epoch = int(47000/64) + 1  , epochs = 50, validation_data = validation_generator, callbacks = callbacks)
model2.save('model4.h5')