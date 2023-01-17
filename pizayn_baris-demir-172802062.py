# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



# Keras Libraries

import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img
mainDIR = os.listdir('../input/piza/piza')

print(mainDIR)
train_folder= '../input/piza/piza/train/'

val_folder = '../input/piza/piza/val/'

test_folder = '../input/piza/piza/test/'
# train 

os.listdir(train_folder)

train_wallet = train_folder+'WALLET/'

train_phone = train_folder+'PHONE/'
from keras import layers

from keras import models



model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
from keras.preprocessing.image import ImageDataGenerator



# All images will be rescaled by 1./255

train_datagen = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_folder,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=5,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        val_folder,

        target_size=(150, 150),

        batch_size=5,

        class_mode='binary')
for data_batch, labels_batch in train_generator:

    print('data batch shape:', data_batch.shape)

    print('labels batch shape:', labels_batch.shape)

    break
history = model.fit_generator(

      train_generator,

      steps_per_epoch=8,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=2)
model.save('wallet_phone_small_1.h5')

import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()
from keras import layers

from keras import models

from keras.regularizers import l2





model2= models.Sequential()

model2.add(layers.Conv2D(32,(3, 3),kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),activation='relu',

                        input_shape=(150, 150, 3)))

model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Conv2D(64, (3, 3),kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),activation='relu'))

model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Conv2D(128, (3, 3),kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),activation='relu'))

model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Conv2D(128, (3, 3),kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),activation='relu'))

model2.add(layers.MaxPooling2D((2, 2)))

model2.add(layers.Flatten())

model2.add(layers.Dense(512,kernel_regularizer=l2(0.000001), bias_regularizer=l2(0.000001),activation='relu'))

model2.add(layers.Dense(1, activation='sigmoid'))
from keras import optimizers



model2.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
history2= model2.fit_generator(

      train_generator,

      steps_per_epoch=8,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=2)
import matplotlib.pyplot as plt





original_model_loss= history.history['val_loss']

l2_regularized_model_loss = history2.history['val_loss']



epochs = range(len(acc))





plt.plot(epochs, original_model_loss, 'b', label='Original model loss')

plt.plot(epochs, l2_regularized_model_loss, 'r', label='l2_regularized_model loss')

plt.title('Original Model and l2_regularized_model Loss')

plt.legend()



plt.figure()





plt.show()
datagen = ImageDataGenerator(

      rotation_range=40,

      width_shift_range=0.2,

      height_shift_range=0.2,

      shear_range=0.2,

      zoom_range=0.2,

      horizontal_flip=True,

      fill_mode='nearest')
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',

                        input_shape=(150, 150, 3)))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))

model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(512, activation='relu'))

model.add(layers.Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer=optimizers.RMSprop(lr=1e-4),

              metrics=['acc'])
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,)



# Note that the validation data should not be augmented!

test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

        # This is the target directory

        train_folder,

        # All images will be resized to 150x150

        target_size=(150, 150),

        batch_size=10,

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        val_folder,

        target_size=(150, 150),

        batch_size=10,

        class_mode='binary')



history = model.fit_generator(

      train_generator,

      steps_per_epoch=8,

      epochs=20,

      validation_data=validation_generator,

      validation_steps=2)
model.save('wallet_and_phone_small_2.h5')

acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()



plt.figure()



plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()



plt.show()