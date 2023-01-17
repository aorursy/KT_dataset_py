import sys

from os.path import join

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import cv2

import tensorflow as tf

from tensorflow.keras.applications.resnet50 import preprocess_input

from tensorflow.keras.preprocessing.image import load_img, img_to_array

#from tensorflow.python.keras.applications import ResNet50



from keras import models, regularizers, layers, optimizers, losses, metrics

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils, to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing import image

from keras.applications import ResNet50

from keras.models import Model,load_model

import itertools

from keras.applications.vgg16 import preprocess_input

import os

print(os.listdir("../input/indian-food/food"))
PATH = "../input/indian-food/food/"

print(os.listdir("../input/indian-food/food/"))
PATHtrain = PATH + 'train/'

print(len(os.listdir(PATHtrain)), " TRAIN Directories of photos")

Labels = os.listdir(PATHtrain)

sig = 0

for label in sorted(Labels):

    print(label,len(os.listdir(PATHtrain + label +'/')))

    sig = sig + len(os.listdir(PATHtrain + label +'/'))



print("Total TRAIN photos ", sig)

print("_"*50)



PATHvalid = PATH + 'test/'

print(len(os.listdir(PATHvalid)), " VALID Directories of photos")

Labels = os.listdir(PATHvalid)

sig = 0

for label in sorted(Labels):

    print(label,len(os.listdir(PATHvalid + label +'/')))

    sig = sig + len(os.listdir(PATHvalid + label +'/'))



print("Total Validation photos ", sig)

print("_"*50)



PATHtest = PATH + 'test/'

print(len(os.listdir(PATHtest)), " TEST Directories of photos")

Labels = os.listdir(PATHtest)

sig = 0

for label in sorted(Labels):

    print(label,len(os.listdir(PATHtest + label +'/')))

    sig = sig + len(os.listdir(PATHtest + label +'/'))



print("Total Testing photos ", sig)

print("_"*50)
conv_base = ResNet50(weights='imagenet',

include_top=False,

input_shape=(224, 224, 3))



print(conv_base.summary())
model = models.Sequential()

model.add(conv_base)

model.add(Conv2D(512, kernel_size=(3, 3),activation='relu',padding='same'))

model.add(MaxPooling2D((3, 3),padding='same'))



model.add(layers.Flatten())

model.add(layers.Dense(1024, activation='relu',kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dense(256, activation='sigmoid',kernel_regularizer=regularizers.l2(0.001)))

model.add(layers.Dense(10, activation='softmax'))



print(model.summary())
for layer in conv_base.layers[:]:

    layer.trainable = False



print('conv_base is now NOT trainable')
model.compile(optimizer=optimizers.Adam(lr=1e-4),

              loss='binary_crossentropy',

              metrics=['accuracy'])



print("model compiled")

print(model.summary())
train_dir = PATHtrain

validation_dir = PATHvalid

test_dir = PATHtest

batch_size = 20

target_size=(224, 224)



#train_datagen = ImageDataGenerator(rescale=1./255)

train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=40,

                                   width_shift_range=0.2,

                                   height_shift_range=0.2,

                                   shear_range=0.2,

                                   zoom_range=0.2,

                                   horizontal_flip=False,

                                   vertical_flip=False,

                                   fill_mode='nearest')



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    train_dir,target_size=target_size,batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(

    validation_dir,target_size=target_size,batch_size=batch_size)

test_generator = test_datagen.flow_from_directory(

    test_dir,target_size=target_size,batch_size=batch_size)
history = model.fit_generator(train_generator,

                              epochs=20,

                              steps_per_epoch = 1077 // batch_size,

                              validation_data = validation_generator,

                              validation_steps = 196 // batch_size)
acc = history.history['accuracy']

val_acc = history.history['val_accuracy']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')

plt.plot(epochs, val_acc, 'r', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'r', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
test_generator = test_datagen.flow_from_directory(

    test_dir,target_size=target_size,batch_size=1)

test_loss, test_acc = model.evaluate_generator(test_generator, steps= 196 // 1, verbose=1)

print('test accuracy:', test_acc)