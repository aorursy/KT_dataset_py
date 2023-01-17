# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os, cv2, re, random

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D

from keras.layers import Activation, Dropout, Flatten, Dense

from keras import backend as K





img_width, img_height = 224, 224



train_data_dir = '../input/train'

validation_data_dir = '../input/test'

nb_train_samples = 500

nb_validation_samples = 100

epochs = 10

batch_size = 32



if K.image_data_format() == 'channels_first':

    input_shape = (3, img_width, img_height)

else:

    input_shape = (img_width, img_height, 3)



input_shape
model = Sequential()

model.add(Conv2D(32, (2, 2), input_shape = input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size =(2, 2)))



model.add(Conv2D(32, (2, 2)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size =(2, 2)))



model.add(Conv2D(64, (2, 2)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size =(2, 2)))

model.add(Dropout(0.4))



model.add(Conv2D(128, (2, 2)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size =(2, 2)))



model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(12))

model.add(Activation('sigmoid'))



model.compile(loss ='binary_crossentropy',optimizer ='adadelta', metrics =['accuracy'])



train_datagen = ImageDataGenerator(rescale = 1. / 255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1. / 255)



train_generator = train_datagen.flow_from_directory(train_data_dir,target_size =(img_width, img_height),

                                                    batch_size = batch_size, class_mode ='categorical')



validation_generator = test_datagen.flow_from_directory(validation_data_dir,target_size =(img_width, img_height),

                                                        batch_size = batch_size, class_mode ='categorical')



model.fit_generator(train_generator, steps_per_epoch = nb_train_samples // batch_size,

                    epochs = epochs, validation_data = validation_generator,

                    validation_steps = nb_validation_samples // batch_size)


