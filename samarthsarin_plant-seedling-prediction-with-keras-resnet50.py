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
from keras.models import Sequential

from keras.layers import Dense,Dropout,BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.resnet50 import ResNet50,preprocess_input

import cv2

import matplotlib.pyplot as plt

from keras import backend as K

%matplotlib inline
train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,vertical_flip=True,shear_range=0.2,zoom_range=0.2,

                                  validation_split=0.2,preprocessing_function=preprocess_input)
train_generator = train_datagen.flow_from_directory('../input/v2-plant-seedlings-dataset/nonsegmentedv2/',batch_size=32,target_size = (224,224),class_mode='categorical',subset='training')

test_generator = train_datagen.flow_from_directory('../input/v2-plant-seedlings-dataset/nonsegmentedv2/',batch_size=32,target_size = (224,224),class_mode='categorical',subset='validation')
K.set_learning_phase(0)

base_model = ResNet50(weights = 'imagenet',include_top = False,input_shape = (224,224,3),pooling='avg')

K.set_learning_phase(1)
model = Sequential()

model.add(base_model)

model.add(BatchNormalization())

model.add(Dense(256,activation = 'relu'))

model.add(Dropout(0.2))

model.add(Dense(12,activation = 'softmax'))

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit_generator(train_generator,steps_per_epoch=1000,epochs = 3,validation_data=test_generator,validation_steps=64)
values = history.history
val_loss = values['val_loss']

val_acc = values['val_acc']

loss = values['loss']

acc = values['acc']

epochs = range(3)
plt.plot(epochs,val_loss,label = 'Validation Loss')

plt.plot(epochs,loss,label = 'Training Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
plt.plot(epochs,val_acc,label = 'Validation Accuracy')

plt.plot(epochs,acc,label = 'Training Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend()

plt.show()