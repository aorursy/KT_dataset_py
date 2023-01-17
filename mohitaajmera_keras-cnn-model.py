#Getting required libraries

import os

import pandas as pd

import numpy as np

from PIL import Image

import cv2

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.preprocessing.image import img_to_array

from sklearn.model_selection import train_test_split

from sklearn import preprocessing
#Fixing Image size 

IMG_SIZE = 128

    
import pickle 

pickle_in = open("../input/xpickle/X.pickle","rb")

X = pickle.load(pickle_in)

pickle_in = open("../input/ypickle/y.pickle","rb")

y = pickle.load(pickle_in)
X = np.array(X)
from keras.utils import np_utils



y = np_utils.to_categorical(y)

(x_train, x_test, y_train, y_test) = train_test_split(X,y, test_size = .2, random_state = 121)



#X = X/255.0

model = Sequential()



model.add(Conv2D(32, (3, 3), input_shape=X.shape[1:],activation = 'relu'))

model.add(Conv2D(32, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(BatchNormalization())

#model.add(Dropout(0.2))



model.add(Conv2D(64, (3, 3),activation = 'relu'))

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(BatchNormalization())

#model.add(Dropout(0.2))



model.add(Conv2D(128, (3, 3), activation ='relu'))

model.add(Conv2D(128, (3, 3), activation ='relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

#model.add(Dropout(.2))



model.add(Flatten())



model.add(Dense(128, activation = 'relu'))

model.add(Dense(12, activation = 'softmax'))



model.compile(optimizer ='adam', loss='categorical_crossentropy',

              metrics=['accuracy'])



model.fit(x_train, y_train,

          epochs=20,

          batch_size=128)

score = model.evaluate(x_test, y_test, batch_size=128)



#model.fit(X, y, batch_size=64, epochs=10, validation_split=0.2 )





#Epoch 20/20

#4431/4431 [==============================] - 10s 2ms/step - loss: 0.0529 - acc: 0.9824
print(score)