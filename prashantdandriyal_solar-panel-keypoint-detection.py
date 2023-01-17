from keras.layers import Conv2D,Dropout,Dense,Flatten

from keras.models import Sequential

from keras.layers.advanced_activations import LeakyReLU

from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from IPython.display import clear_output

from time import sleep

import os

import pandas as pd

from PIL import Image

import os, sys, cv2

# For X

X = []

img_path = '../input/solarpanels308rgb/rgb_proc_imgs/'

dirs = os.listdir(img_path)



#The gem

dirs = sorted(dirs)



for item in dirs:

    if os.path.isfile(img_path+item):

        img = cv2.imread(img_path+item)

        # Read & Append to feature set

        #print(type(img))

        X.append(img)



X_train = np.array(X,dtype = 'float')

#print(type(X_train))

X_train = X_train.reshape(-1,480,480,3)
# for y

y = []

df = pd.read_csv('../input/solarpanels308csv/parsed_df.csv')

ddf = df.drop('img_name',axis=1)

y_train = ddf.to_numpy()
# Create models

model = Sequential()



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(480,480,3)))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

# model.add(BatchNormalization())

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())

model.add(MaxPool2D(pool_size=(2, 2)))



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())



model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

model.add(LeakyReLU(alpha = 0.1))

model.add(BatchNormalization())





model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(8))

model.summary()

modelC = model



model.compile(optimizer='adam',

              loss='mean_squared_error',

              metrics=['mae'])

logC = modelC.fit(X_train,y_train,epochs = 50,batch_size = 8,validation_split = 0.2)
# Plotting loss and accuracy curves for training and verification

fig, ax = plt.subplots(2,1)

log = logC

# accuracy

ax[0].plot(log.history['mae'], color='b', label="Training MAE")

ax[0].plot(log.history['val_mae'], color='r',label="Validation MAE")

legend = ax[0].legend(loc='best', shadow=True)



# loss

ax[1].plot(log.history['loss'], color='b', label="Training loss")

ax[1].plot(log.history['val_loss'], color='r', label="validation loss",axes =ax[1])

legend = ax[1].legend(loc='best', shadow=True)

fig.show()
#from google.colab.patches import cv2_imshow

import matplotlib.pyplot as plt

import cv2

def pred(test_img):

  res = modelC.predict(test_img.reshape(1,480,480,3))

  #test_img = np.float32(test_img)

  cv2.circle(test_img, (res[0][0], res[0][1]), 5, (0,255,0), -1)

  cv2.circle(test_img, (res[0][2], res[0][3]), 5, (0,255,0), -1)

  cv2.circle(test_img, (res[0][4], res[0][5]), 5, (0,255,0), -1)

  cv2.circle(test_img, (res[0][6], res[0][7]), 5, (0,255,0), -1)

  #cv2.imshow('yoo.png',test_img)

  #cv2.imwrite('yo.png', test_img)

  plt.imshow(test_img)
test_img = cv2.imread("../input/solarpanels308rgb/rgb_proc_imgs/00000071_rgb_resized.jpg")

plt.imshow(test_img)



#predict

pred(test_img)