import matplotlib.pyplot as plt 

import os

from PIL import Image

import pandas as pd

import numpy as np

import cv2

#print(os.listdir("../input"))



import keras

from keras.models import Sequential

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dropout

from keras.layers import Flatten

from keras.layers import Dense

from keras.preprocessing.image import ImageDataGenerator, load_img
Train = 'chest_xray/chest_xray/train'

Test = 'chest_xray/chest_xray/test'

Val = 'chest_xray/chest_xray/val'
training_data = []

CATEGORIES = ['NORMAL', 'PNEUMONIA']

IMG_SIZE  = 150
os.listdir()
for category in CATEGORIES:

        path = os.path.join(Train, category)

        label = CATEGORIES.index(category)

        for img in os.listdir(path):

                try:

                    img_array = cv2.imread(os.path.join(path, img))

                    new_arr = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    training_data.append([label, new_arr])

                except Exception as e:

                    pass 



for category in CATEGORIES:

        path = os.path.join(Test, category)

        label = CATEGORIES.index(category)

        for img in os.listdir(path):

                try:

                    img_array = cv2.imread(os.path.join(path, img))

                    new_arr = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    training_data.append([label, new_arr])

                except Exception as e:

                    pass 

                

for category in CATEGORIES:

        path = os.path.join(Val, category)

        label = CATEGORIES.index(category)

        for img in os.listdir(path):

                try:

                    img_array = cv2.imread(os.path.join(path, img))

                    new_arr = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))

                    training_data.append([label, new_arr])

                except Exception as e:

                    pass 
import random 
random.shuffle(training_data)
X = []

y = []
len(training_data)
for i in training_data:

    X.append(i[1])

    y.append(i[0])
X = np.array(X) 
X = X/255.0
model = Sequential()

model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', input_shape = (150,150,3), activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

          

model.add(Conv2D(64, (3,3), strides=(1, 1), padding='valid', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.2))

          

model.add(Flatten())

model.add(Dense(64, activation='relu'))

          

model.add(Dense(1, activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

          

model.fit(X,y, batch_size=32, epochs=20, validation_split=0.2)