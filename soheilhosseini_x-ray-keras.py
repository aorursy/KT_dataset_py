# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from pathlib import Path



# Path to train directory 

train_directory = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/train') 



# Path to validation directory

val_directory = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/val')  



# Path to test directory

test_directory = Path('../input/chest-xray-pneumonia/chest_xray/chest_xray/test')
from keras.preprocessing.image import ImageDataGenerator



# create a data generator

datagen = ImageDataGenerator(rescale=1./255)
# load and iterate training dataset

train_it = datagen.flow_from_directory(train_directory, class_mode='binary', batch_size=64)

# load and iterate validation dataset

val_it = datagen.flow_from_directory(val_directory, class_mode='binary', batch_size=64)

# load and iterate test dataset

test_it = datagen.flow_from_directory(test_directory, class_mode='binary', batch_size=64)
from keras import layers

from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(256, 256, 3)))

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
#compile model using accuracy to measure model performance

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
#train the model

model.fit_generator(train_it, steps_per_epoch=10, validation_data=val_it, validation_steps=50,epochs=20)
test_datagen = ImageDataGenerator(rescale=1./255)



test_generator = test_datagen.flow_from_directory(

test_directory,

target_size=(256, 256),

batch_size=64,

class_mode='binary')

test_loss, test_acc = model.evaluate_generator(test_generator, steps=10, verbose=1)

print('test acc:', test_acc)