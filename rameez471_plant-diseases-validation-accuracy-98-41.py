# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout,MaxPool2D,BatchNormalization

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau,ModelCheckpoint,EarlyStopping
classifier=Sequential()



classifier.add(Conv2D(filters=96,kernel_size=(11,11),strides=(4,4),padding='valid',input_shape=(224,224,3),activation='relu'))

classifier.add(MaxPool2D((2,2),strides=(2,2),padding='valid'))

classifier.add(BatchNormalization())



classifier.add(Conv2D(filters=256,kernel_size=(5,5),strides=(1,1),padding='valid',activation='relu'))

classifier.add(MaxPool2D((2,2),strides=(2,2),padding='valid'))

classifier.add(BatchNormalization())



classifier.add(Conv2D(filters=512,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))

classifier.add(BatchNormalization())



classifier.add(Conv2D(filters=1024,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))

classifier.add(BatchNormalization())



classifier.add(Conv2D(filters=1024,kernel_size=(3,3),strides=(1,1),padding='valid',activation='relu'))



classifier.add(MaxPool2D((2,2),strides=(2,2),padding='valid'))

classifier.add(BatchNormalization())



classifier.add(Flatten())



classifier.add(Dense(4096,activation='relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(4096,activation='relu'))

classifier.add(Dropout(0.4))

classifier.add(BatchNormalization())

classifier.add(Dense(1000,activation='relu'))

classifier.add(Dropout(0.2))

classifier.add(BatchNormalization())

classifier.add(Dense(38,activation='softmax'))



classifier.summary()
classifier.compile(optimizer='adam',

                  loss='categorical_crossentropy',

                  metrics=['accuracy'])
train_datagen=ImageDataGenerator(rescale=1./255,

                                shear_range=0.2,

                                zoom_range=0.2,

                                width_shift_range=0.2,

                                height_shift_range=0.2,

                                fill_mode='nearest')



valid_datagen=ImageDataGenerator(rescale=1./255)



batch_size=128

base_dir='../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)'



training_set=train_datagen.flow_from_directory(base_dir+'/train',

                                              target_size=(224,224),

                                              batch_size=batch_size,

                                              class_mode='categorical')



valid_set=valid_datagen.flow_from_directory(base_dir+'/valid',

                                          target_size=(224,224),

                                          batch_size=batch_size,

                                          class_mode='categorical')
print(training_set.class_indices)
train_num=training_set.samples

valid_num=valid_set.samples

reduce_lr=ReduceLROnPlateau(factor=0.5,patience=5)



weights=ModelCheckpoint('weights.hdf5',

                       save_best_only=True,

                       verbose=1,

                       save_weights_only=True)



early_stopping=EarlyStopping(monitor='val_acc',mode='max',patience=5)



history=classifier.fit_generator(training_set,

                                steps_per_epoch=256,

                                validation_data=valid_set,

                                epochs=50,

                                validation_steps=128,

                                callbacks=[reduce_lr,weights,early_stopping])
