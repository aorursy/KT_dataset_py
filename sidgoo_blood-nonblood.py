# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



os.listdir('/kaggle/input')

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import matplotlib.image as im



#for i 

img  = im.imread("/kaggle/input/TrainingImagesold/Blood/TCGA-06-0213-01Z-00-DX3.337437a8-ffca-4ca2-9b94-902ce032d5f4id-5ae351f192ca9a0020d95f9b.jpg")

plt.imshow(img)

img.shape
from PIL import Image

img  = Image.open('/kaggle/input/TrainingImagesold/Blood/TCGA-06-0213-01Z-00-DX3.337437a8-ffca-4ca2-9b94-902ce032d5f4id-5ae351f192ca9a0020d95f9b.jpg')

img
import keras

from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D,Activation,Dropout

from keras.models import Sequential

from keras.preprocessing.image import ImageDataGenerator
batchsize = 10



data_generator = ImageDataGenerator(rescale=1.0/255)

train_data = data_generator.flow_from_directory('/kaggle/input/TrainingImagesold',target_size=(150,150),batch_size=batchsize,class_mode='binary')

test_data = data_generator.flow_from_directory('/kaggle/input/TestingImagesold',target_size=(150,150),batch_size=batchsize,class_mode='binary')
for i in train_data:

    print(i[0].shape)

    break
for i in train_data:

    print(i[1])

    break
model = Sequential()

model.add(Conv2D(32,(5,5),input_shape=(150,150,3),activation='relu'))

model.add(MaxPooling2D(2,2))



model.add(Conv2D(64,(5,5),activation='relu'))

model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(32,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit_generator(train_data,epochs=10,validation_data=test_data,verbose=1,steps_per_epoch=12,validation_steps=12,shuffle=True)