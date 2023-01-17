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

        os.path.join(dirname, filename)



# Any results you write to the current directory are saved as output.
from keras.preprocessing.image import ImageDataGenerator 



datagen = ImageDataGenerator(validation_split=0.2)
train = datagen.flow_from_directory('/kaggle/input/bangla-ocr/Train/', class_mode='categorical', color_mode = 'grayscale', target_size = (32,32), batch_size=64, subset='training')
val = datagen.flow_from_directory('/kaggle/input/bangla-ocr/Train/', class_mode='categorical', color_mode = 'grayscale', target_size = (32,32), batch_size=64, subset='validation') 
from keras.models import Sequential 

from keras.layers import Conv2D, MaxPooling2D 

from keras.layers import Activation, Dropout, Flatten, Dense 

from keras import backend as K 

from glob import glob

from keras.preprocessing.image import ImageDataGenerator

from keras.layers import Convolution2D,MaxPooling2D,Dropout,Dense,Flatten,BatchNormalization,Conv2D

from keras.applications.resnet50 import ResNet50
model1 = Sequential()

model1.add(Convolution2D(32,(3,3),activation='relu',input_shape = (32,32,1)))

model1.add(MaxPooling2D(2,2))

model1.add(Convolution2D(64,(3,3),activation='relu'))

model1.add(BatchNormalization())

model1.add(MaxPooling2D(2,2))

model1.add(Dropout(0.2))

model1.add(Convolution2D(128,(3,3),activation='relu'))

model1.add(BatchNormalization())

model1.add(MaxPooling2D(2,2))

model1.add(Dropout(0.2))

model1.add(Flatten())

model1.add(Dense(256,activation='relu'))

model1.add(Dense(512,activation='relu'))

model1.add(Dense(100,activation='relu'))

model1.add(Dense(11,activation='softmax'))

model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model1.summary()
model1.fit_generator(train,steps_per_epoch=500,

                              epochs = 50,validation_data=val,validation_steps=40)
test = datagen.flow_from_directory('/kaggle/input/bangla-ocr/Test/',color_mode='grayscale', class_mode='categorical', target_size = (32,32), batch_size=64)
model1.save("model_gray2.h5")

print("Saved model to disk")