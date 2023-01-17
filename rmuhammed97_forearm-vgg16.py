import os

import keras

import tflearn

import tensorflow as tf

import cv2

import numpy as np

import matplotlib.pyplot as plt

import joblib

import pandas as pd

from tqdm import tqdm

from random import shuffle

from sklearn.preprocessing import MinMaxScaler, scale

from sklearn.model_selection import train_test_split

from tflearn.layers.conv import conv_2d, max_pool_2d

from tflearn.layers.core import input_data, dropout, fully_connected, flatten

from tflearn.layers.estimator import regression

from tflearn.metrics import Accuracy

from keras.layers import Dense

from keras.models import Sequential

from keras.optimizers import Adam, SGD

from keras.layers import Conv2D, Activation, ZeroPadding2D, MaxPooling2D, Flatten, Dropout
FORE_DIR = '../input/xr_forearm_train/XR_FOREARM_TRAIN'

MODEL_NAME = 'ForearmAbnormalityDetection'

IMG_SIZE = 224



def LoadTrain():

    train_data = []

    for root, dirs, files in os.walk(FORE_DIR):

        for name in files:

            if name.endswith((".png", ".jpg",".jpeg")):

                path = os.path.join(root,name)

                img = cv2.imread(path,0) # Reading Image

                if img is None: continue

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)) # Adaptive Histogram Equalization Object

                adaptedimage = clahe.apply(img)

                adaptedimage = np.divide(adaptedimage,255) # Image Normalization

                adaptedimage = cv2.resize(adaptedimage,(IMG_SIZE,IMG_SIZE)) # resizing Image

                

                rows,cols = adaptedimage.shape

                M = cv2.getRotationMatrix2D((cols/2,rows/2),45,1)

                rotated_img = adaptedimage.copy()

                rotated_img = cv2.warpAffine(rotated_img,M,(cols,rows))

                

                M2 = cv2.getRotationMatrix2D((cols/2,rows/2),135,1)

                rotated_img2 = adaptedimage.copy()

                rotated_img2 = cv2.warpAffine(rotated_img2,M2,(cols,rows))

                

                vertical_img = adaptedimage.copy()

                vertical_img = cv2.flip(vertical_img,1)

                blurred_img = adaptedimage.copy()

                blurred_img = cv2.blur(blurred_img,(5,5))

                labely = np.zeros((2))

                if 'positive' in root: labely[1] = 1

                elif 'negative' in root: labely[0] = 1

                train_data.append([np.array(adaptedimage),labely])

                train_data.append([np.array(vertical_img),labely])

                train_data.append([np.array(blurred_img),labely])

                #train_data.append([np.array(rotated_img),labely])

                #train_data.append([np.array(rotated_img2),labely])

    shuffle(train_data)

    print ('train data images readed Successfuly!')

    return train_data
train_data = LoadTrain()



X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print ('train Images Loaded Succesfully')

y = np.array([i[1] for i in train_data])

print ('train labels Loaded Succesfully')
print (X.shape)

print (y.shape)
model = Sequential()

model.add(ZeroPadding2D((1, 1), input_shape=X.shape[1:]))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(64, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(128, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(256, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(ZeroPadding2D((1, 1)))

model.add(Conv2D(512, (3, 3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(Flatten())

model.add(Dense(4096))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(4096))

model.add(Activation('relu'))

model.add(Dropout(0.5))

model.add(Dense(2))

model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, nesterov=True)



model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(X, y, validation_split=0.2, epochs=101)

model.save('forearm VGG16.h5')
FORE_DIR_TEST = '../input/xr_forearm_valid/XR_FOREARM_VALID'

IMG_SIZE = 224



def LoadTest():

    test_data = []

    for root, dirs, files in os.walk(FORE_DIR_TEST):

        print (root)

        for name in files:

            if name.endswith((".png", ".jpg",".jpeg")):

                path = os.path.join(root,name)

                img = cv2.imread(path,0) # Reading Image

                img = np.divide(img,255) # Image Normalization

                img = cv2.resize(img,(IMG_SIZE,IMG_SIZE)) # resizing Image

                if img is None: continue

                labely = np.zeros((2))

                if 'positive' in root: labely[1] = 1

                elif 'negative' in root: labely[0] = 1

                test_data.append([np.array(img),labely])

    print ('test data images readed Successfuly!')

    return test_data
test_data = LoadTest()



X_test = np.array([i[0] for i in test_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

print ('test Images Loaded Succesfully')

y_test = np.array([i[1] for i in test_data])

print ('test labels Loaded Succesfully')



predictions = model.predict(X_test)
kt = 0

for i in range(np.array(predictions).shape[0]):

    mxindx = np.argmax(predictions[i,:])

    mxindx2 = np.argmax(y_test[i,:])

    if mxindx==mxindx2: kt+=1

acc = kt/(np.array(predictions).shape[0])

print ('Test accurcy: {}%'.format(acc))