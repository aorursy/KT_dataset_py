import numpy as np

#import tensorflow as tf

#import keras as k

import pandas as pd

import matplotlib.pyplot as plt

import cv2 

import math

import os
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

fps = 30

title = 'normal speed video'

delay = int(100/ fps)

X = []

count = 0

for i in range (1,1000):

    videoFile = "/kaggle/input/real-life-violence-situations-dataset/real life violence situations/Real Life Violence Dataset/Violence/V_%d.mp4" % i

    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path

    frameRate = cap.get(5) #frame rate

    #print(cap.isOpened())

    while(cap.isOpened()):

        #print(1)

        frameId = cap.get(1) #current frame number

        ret, frame = cap.read()

        if (ret != True):

            break

        if (frameId % math.floor(frameRate) == 0):

            X_temp = cv2.resize(frame, (64,64))

            X.append(X_temp)

    cap.release()
#1 for violence





X = np.reshape(X, (5827, 64*64*3))

print(np.shape(X))

X = np.concatenate((X,np.ones((5827,1))), axis = 1)

print(np.shape(X))
X2 = []

count = 0

for i in range (1,1000):

    videoFile = "/kaggle/input/real-life-violence-situations-dataset/real life violence situations/Real Life Violence Dataset/NonViolence/NV_%d.mp4" % i

    cap = cv2.VideoCapture(videoFile)   # capturing the video from the given path

    frameRate = cap.get(5) #frame rate

    #print(frameRate)

    while(cap.isOpened()):

        frameId = cap.get(1) #current frame number

        ret, frame = cap.read()

        if (ret != True):

            break

        if (frameId % math.floor(frameRate) == 0):

            X_temp = cv2.resize(frame, (64,64))

            X2.append(X_temp)

    cap.release()
print(np.shape(X2))
#0 for non-violence





X2 = np.reshape(X2, (4980, 64*64*3))

print(np.shape(X2))

X2 = np.concatenate((X2,np.zeros((4980,1))), axis = 1)

print(np.shape(X2))

X_true = np.concatenate((X,X2), axis = 0)

print(np.shape(X_true))
np.random.shuffle(X_true)

X_true = X_true.astype(int)

#print(X_true)
y_true = X_true[:, -1]

print(y_true)

X_true = np.delete(X_true, -1, 1)
from keras.models import Sequential

from keras.layers import Dense, Activation, Dropout

from keras import optimizers

import keras as K

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

seed = 78

test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X_true, y_true, test_size=test_size, random_state=seed)
from numpy import loadtxt

from xgboost import XGBClassifier
model = XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]

y_pred = model.predict(X_test).round()
# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
np.savetxt('Desktop/Y_true.csv', y_true, delimiter=',')
np.savetxt('Desktop/X_true.csv', X_true, delimiter=',')
import math

import numpy as np

import h5py

import matplotlib.pyplot as plt

import scipy

from PIL import Image

from scipy import ndimage

import tensorflow as tf

from tensorflow.python.framework import ops

import keras as k

from keras.layers.pooling import AveragePooling2D

import cv2

import math

from keras.layers import Dense, Conv2D, Flatten

from keras.models import Sequential

from keras.layers import Dense, Activation

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
#CNN begins here
X_train = np.reshape(X_train, (7240,64,64,3))
X_test = np.reshape(X_test, (3567,64,64,3))
model = Sequential()



model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(64,64,3)))

K.layers.GlobalAveragePooling1D(data_format='channels_last')

model.add(Conv2D(32, kernel_size=3, activation='sigmoid'))

K.layers.GlobalAveragePooling1D(data_format='channels_last')

model.add(Conv2D(16, kernel_size=3, activation='sigmoid'))

K.layers.GlobalAveragePooling1D(data_format='channels_last')

model.add(Flatten())

model.add(Dense(1600, activation='softmax'))

model.add(Dense(80, activation='softmax'))

model.add(Dense(14, activation='softmax'))

model.add(Dense(1, activation='softmax'))

model.summary()
K.optimizers.Nadam(learning_rate=0.0002, beta_1=0.9, beta_2=0.999)

model.compile(optimizer = 'nadam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=10,batch_size=32)