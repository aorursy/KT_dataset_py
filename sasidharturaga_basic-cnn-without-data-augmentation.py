import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras import backend as K

from keras.preprocessing import image

from keras.applications.mobilenet import MobileNet

from keras.applications.vgg16 import preprocess_input, decode_predictions

from keras.models import Model

import timeit

import pylab as plt

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

# Reading Data

d=pd.read_csv('../input/digit-recognizer/train.csv')

l=d.iloc[:,0]

tr=d.iloc[:,1:]

tr=tr.as_matrix()

# Reshaping the data so the it can be formatted according to the tensorflow requirements

train_d,test_d,train_l,test_l=train_test_split(tr,l,test_size=0.33)

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':

    train_d =  train_d.reshape( train_d.shape[0], 1, img_rows, img_cols)

    test_d = test_d.reshape(test_d.shape[0], 1, img_rows, img_cols)

    input_shape = (1, img_rows, img_cols)

else:

    train_d =  train_d.reshape( train_d.shape[0], img_rows, img_cols, 1)

    test_d = test_d.reshape(test_d.shape[0], img_rows, img_cols, 1)

    input_shape = (img_rows, img_cols, 1)

train_d = train_d.astype('float32')

test_d = test_d.astype('float32')

train_d/=255

test_d/=255

train_l = keras.utils.to_categorical(train_l, 10)

test_l = keras.utils.to_categorical(test_l, 10)

# Creating Model

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))

model.summary()
# Training the model

model.compile(loss=keras.losses.categorical_crossentropy,

              optimizer=keras.optimizers.Adadelta(),

              metrics=['accuracy'])

model.fit(train_d, train_l,batch_size=128,epochs=50,verbose=1,validation_data=(test_d, test_l))
# Testing the model and making Prediction

t=pd.read_csv('../input/digit-recognizer/test.csv')

t = t.as_matrix()

if K.image_data_format() == 'channels_first':

     t =  t.reshape(t.shape[0], 1, img_rows, img_cols)

else:

     t = t.reshape(t.shape[0], img_rows, img_cols, 1)

t = t.astype('float32')

t /= 255

prediction = model.predict(t)



result = np.argmax(prediction,axis=1)
np.savetxt('cnn9931.csv',np.c_[range(1,len(t) + 1),result],delimiter=',',header='ImageId,Label', comments='',fmt='%d')