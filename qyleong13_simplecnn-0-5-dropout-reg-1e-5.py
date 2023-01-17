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
import keras

keras.__version__
from keras import models

from keras import layers

from keras import regularizers

from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout

from keras.layers.normalization import BatchNormalization

model = models.Sequential()

model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(300,300,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(BatchNormalization())

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dropout(0.5))

model.add(layers.Dense(256,kernel_regularizer=regularizers.l2(1e-5), activation='relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1,activation='sigmoid'))

model.summary()
import os

train_dir = "../input/kaggle_dataset/kaggle_dataset/train"

validation_dir = "../input/kaggle_dataset/kaggle_dataset/validation"

test_dir ="../input/kaggle_dataset/kaggle_dataset/test"

print('total train CXR(benign): ', len(os.listdir(train_dir+"/benign")))

print('total train CXR(malignant): ', len(os.listdir(train_dir+"/malignant")))

print('total validation CXR(benign): ',len(os.listdir(validation_dir+"/benign")))

print('total validation CXR(malignant):', len(os.listdir(validation_dir+"/malignant")))

print('total test CXR(benign): ', len(os.listdir(test_dir+"/benign")))

print('total test CXR(malignant): ', len(os.listdir(test_dir+"/malignant")))
from keras.preprocessing.image import ImageDataGenerator

#Image Preprocessing

train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=40,

        width_shift_range=0.2,

        height_shift_range=0.2,

        shear_range=0.2,

        zoom_range=0.2,

        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(

        

        # This is the target directory

        train_dir,

        # All images will be resized to 300x300

        target_size=(300, 300),

        batch_size=2, # power of 2 is recommended

        # Since we use binary_crossentropy loss, we need binary labels

        class_mode='binary')



validation_generator = test_datagen.flow_from_directory(

        validation_dir,

        target_size=(300, 300),

        batch_size=2,

        class_mode='binary')
from keras import optimizers

model.compile(loss='binary_crossentropy',

              optimizer=optimizers.Adam(lr=1e-4),

             metrics=['acc'])
# checkpoint

from keras.callbacks import ModelCheckpoint

filepath="weights.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint]
history = model.fit_generator(

    train_generator,

    steps_per_epoch=564,

    epochs=200,

    validation_data = validation_generator,

    validation_steps=70,

    callbacks = callbacks_list)
import matplotlib.pyplot as plt

acc= history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')

plt.plot(epochs, val_acc,'b', label='Validation acc')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()
plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
filenames = validation_generator.filenames

nb_samples = len(filenames)

valid_loss, valid_accuracy = model.evaluate_generator(validation_generator,70)

print('valid_accuracy: ', valid_accuracy)
filenames = validation_generator.filenames

nb_samples = len(filenames)

valid_loss, valid_accuracy = model.evaluate_generator(validation_generator,nb_samples)

print('valid_accuracy: ', valid_accuracy)