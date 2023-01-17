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
import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator
tf.__version__
#preprocessing the data using image data generator
train_datagen=ImageDataGenerator(

rescale=1./255,

shear_range=0.2,

zoom_range=0.2,

horizontal_flip=True)
train_set=train_datagen.flow_from_directory(

'/kaggle/input/cat-and-dog/training_set/training_set',

target_size=(128,128),

batch_size=32,class_mode='binary')
test_datagen=ImageDataGenerator(rescale=1./255)



test_set=test_datagen.flow_from_directory(

'/kaggle/input/cat-and-dog/test_set/test_set',

target_size=(128,128),

batch_size=32,class_mode='binary')
#building the CNN
from keras.models import Sequential

from keras.layers import Convolution2D

from keras.layers import MaxPool2D

from keras.layers import Dense

from keras.layers import Flatten
cnn=Sequential()

cnn.add(Convolution2D(32,(3,3),input_shape=(128,128,3),activation='relu'))

cnn.add(MaxPool2D((2,2),strides=2))



cnn.add(Convolution2D(32,(3,3),activation='relu'))

cnn.add(MaxPool2D((2,2),strides=2))

cnn.add(Convolution2D(16,(3,3),activation='relu'))



cnn.add(Flatten())

cnn.add(Dense(128,activation='relu'))

cnn.add(Dense(64,activation='relu'))

cnn.add(Dense(1,activation='sigmoid'))
#compiling the model

cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#fitting the data

cnn.fit(x=train_set,validation_data=test_set,epochs=25)
#predicting
from keras.preprocessing import image

prediction_image=image.load_img("/kaggle/input/cat-and-dog/test_set/test_set/dogs/dog.4016.jpg",target_size=(128,128))

prediction_image=image.img_to_array(prediction_image)

#expanding the dimension beacuse we are fitting the data in model in batches

prediction_image=np.expand_dims(prediction_image,axis=0)


result=cnn.predict(prediction_image)
print(result)