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
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
train_datagen=ImageDataGenerator(rescale=1./255,
                                shear_range=0.2,
                                zoom_range=0.2,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                fill_mode='nearest')

valid_gen=ImageDataGenerator(rescale=1./255)

batch_size=32
base_dir="../input/new-plant-diseases-dataset/new plant diseases dataset(augmented)/New Plant Diseases Dataset(Augmented)"

training_set=train_datagen.flow_from_directory(base_dir+'/train',
                                              target_size=(224,224),
                                              batch_size=batch_size,
                                              class_mode='categorical')

valid_set=valid_gen.flow_from_directory(base_dir+'/valid',
                                       target_size=(224,224),
                                       batch_size=batch_size,
                                       class_mode='categorical')
train_num=training_set.samples
valid_num=valid_set.samples
base_model=tf.keras.applications.VGG16(include_top=False,
                                      weights='imagenet',
                                      input_shape=(224,224,3))
base_model.summary()
base_model.trainable=False
base_model.summary()
inputs=Input(shape=(224,224,3))
x=base_model(inputs,training=False)
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation='relu')(x)
x=Dropout(0.2)(x)
x=Dense(512,activation='relu')(x)
x=Dropout(0.2)(x)
outputs=Dense(38,activation='softmax')(x)


vgg_model=Model(inputs,outputs)

vgg_model.summary()

vgg_model.compile(optimizer=tf.keras.optimizers.Adam(),
             loss='categorical_crossentropy',
             metrics=['accuracy'])
len(vgg_model.trainable_variables)
vgg_model.evaluate(valid_set)
weightpath='best_weights_9.hdf5'
checkpoint = ModelCheckpoint(weightpath, monitor='val_acc', verbose=1, 
                             save_best_only=True, save_weights_only=True, mode='max')

history=vgg_model.fit_generator(training_set,
                        steps_per_epoch=150,
                        epochs=15,
                        validation_data=valid_set,
                        validation_steps=100,
                        callbacks=[checkpoint])
history=vgg_model.fit_generator(training_set,
                        steps_per_epoch=150,
                        epochs=15,
                        validation_data=valid_set,
                        validation_steps=100,
                        callbacks=[checkpoint])
history=vgg_model.fit_generator(training_set,
                        steps_per_epoch=150,
                        epochs=15,
                        validation_data=valid_set,
                        validation_steps=100,
                        callbacks=[checkpoint])
base_model.trainable=True #Un-Freezing the base model
vgg_model.summary()
vgg_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

fine_tune_history=vgg_model.fit_generator(training_set,
                                        steps_per_epoch=150,
                                        validation_data=valid_set,
                                        epochs=10,
                                        validation_steps=100)
