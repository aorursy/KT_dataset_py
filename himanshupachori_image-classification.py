# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/dog vs cat/dataset"))



# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
model = Sequential()
model.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))

model.add(MaxPool2D(2,2))

model.add(Flatten())

model.add(Dense(128,activation='relu'))

model.add(Dense(1,activation='sigmoid')) 
#Compile the network

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['binary_accuracy'])
from keras.optimizers import Adam

from keras.losses import binary_crossentropy

from keras.metrics import binary_accuracy
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/dog vs cat/dataset/training_set',

target_size = (64, 64),

batch_size = 32,

class_mode = 'binary')
test_set = test_datagen.flow_from_directory('../input/dog vs cat/dataset/test_set',

target_size = (64, 64),

batch_size = 32,

class_mode = 'binary')
model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['binary_accuracy'])

model.fit_generator(training_set, 

                    steps_per_epoch = 5000, 

                    epochs = 2,

                    validation_data = test_set,

                    validation_steps = 2500)
#Initiate the classifier

model_2 = Sequential()

model_2.add(Conv2D(32,3,3,input_shape=(64,64,3),activation='relu'))

model_2.add(Conv2D(32, (3, 3), activation = 'relu'))

model_2.add(Conv2D(32, (3, 3), activation = 'relu'))



model_2.add(MaxPool2D(2,2))



model_2.add(Conv2D(32, (3, 3), activation = 'relu'))

model_2.add(MaxPool2D(pool_size = (2, 2)))



model_2.add(Flatten())



model_2.add(Dense(128,activation='relu'))

model_2.add(Dense(128,activation='relu'))

model_2.add(Dense(128,activation='relu'))



model_2.add(Dense(1,activation='sigmoid')) 



model_2.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.2,

                                   zoom_range = 0.2,

                                   horizontal_flip = True)



test_datagen = ImageDataGenerator(rescale = 1./255)



training_set = train_datagen.flow_from_directory('../input/dog vs cat/dataset/training_set',

target_size = (64, 64),

batch_size = 32,

class_mode = 'binary')



test_set = test_datagen.flow_from_directory('../input/dog vs cat/dataset/test_set',

target_size = (64, 64),

batch_size = 32,

class_mode = 'binary')
model_2.fit_generator(training_set,

steps_per_epoch = 5000,

epochs = 2,

validation_data = test_set,

validation_steps = 2500)
print('hello')