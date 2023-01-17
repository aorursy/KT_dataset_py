# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import cv2

from keras.callbacks import ModelCheckpoint

from keras.layers import Conv2D, Flatten, MaxPooling2D,Dense,Dropout

from keras.models  import Sequential

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

import random,os,glob

import matplotlib.pyplot as plt





dir_path = '../input/garbage-classification/Garbage classification/Garbage classification'



img_list = glob.glob(os.path.join(dir_path, '*/*.jpg'))



len(img_list)





train=ImageDataGenerator(horizontal_flip=True, vertical_flip=True,validation_split=0.1,rescale=1./255,

                         shear_range = 0.1,zoom_range = 0.1,

                         width_shift_range = 0.1,

                         height_shift_range = 0.1,)

test=ImageDataGenerator(rescale=1/255,validation_split=0.1)

train_generator=train.flow_from_directory(dir_path,target_size=(300,300),batch_size=32,

                                          class_mode='categorical',subset='training')

test_generator=test.flow_from_directory(dir_path,target_size=(300,300),batch_size=32,

                                        class_mode='categorical',subset='validation')

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())



print(labels)



model=Sequential()

    

model.add(Conv2D(32,(3,3), padding='same',input_shape=(300,300,3),activation='relu'))

model.add(MaxPooling2D(pool_size=2)) 

model.add(Conv2D(64,(3,3), padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2)) 

model.add(Conv2D(32,(3,3), padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=2)) 

model.add(Flatten())

model.add(Dense(64,activation='relu'))

model.add(Dense(6,activation='softmax'))



filepath="trained_model.h5"

checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

callbacks_list = [checkpoint1]



print(model.summary())



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.fit_generator(train_generator, epochs=100, steps_per_epoch=2276//32,validation_data=test_generator,validation_steps=251//32,callbacks=callbacks_list)