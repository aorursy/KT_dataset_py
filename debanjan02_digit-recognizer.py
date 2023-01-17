import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))

from glob import glob

from PIL import Image

%matplotlib inline

import matplotlib.pyplot as plt

import cv2

import fnmatch

import keras

from time import sleep

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization,MaxPooling2D,Activation

from keras.optimizers import RMSprop,Adam

from tensorflow.keras.callbacks import EarlyStopping

from keras import backend as k
df = pd.read_csv('../input/train.csv')

y_train = np.array(df['label'])

x_train = np.array(df.drop(['label'], axis=1))

x_train = x_train.reshape(42000, 28, 28, 1)
df = pd.read_csv('../input/test.csv')

x_check = np.array(df)

x_check = x_check.reshape(28000, 28, 28, 1)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 101)

y_train = to_categorical(y_train, num_classes = 10)

y_test = to_categorical(y_test, num_classes = 10)
import keras

from keras.models import Sequential,Input,Model

from keras.layers import Conv2D, MaxPooling2D, MaxPooling1D, GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, LSTM, TimeDistributed

from keras.layers.normalization import BatchNormalization

from keras.layers.advanced_activations import LeakyReLU

from keras.applications.vgg16 import VGG16

def base_model(x):

    BATCH_NORM = x

    model = Sequential()



    model.add(Conv2D(64, (3, 3), padding='same', input_shape=x_train.shape[1:], name='block1_conv1'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(64, (3, 3), padding='same', name='block1_conv2'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))



    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv1'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(128, (3, 3), padding='same', name='block2_conv2'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))



    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv1'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv2'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv3'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(256, (3, 3), padding='same', name='block3_conv4'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv1'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv2'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv3'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block4_conv4'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv1'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv2'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv3'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Conv2D(512, (3, 3), padding='same', name='block5_conv4'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))



    model.add(Flatten())



    model.add(Dense(4096))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))

    model.add(Dropout(0.5))



    model.add(Dense(4096, name='fc2'))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('relu'))

    model.add(Dropout(0.5))



    model.add(Dense(10))

    model.add(BatchNormalization()) if BATCH_NORM else None

    model.add(Activation('softmax'))



    model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])

    return model

#model.summary()

cnn = base_model(1)
cnn.load_weights('model_check_path_2.hdf5')
from sklearn.metrics import classification_report

pred = cnn.predict(x_test)

print(classification_report(np.argmax(y_test, axis = 1),np.argmax(pred, axis = 1)))
prediction = cnn.predict(x_check)
pred_sub = []

for i in range(len(prediction)):

    pred_sub.append(prediction[i].argmax())

pred_sub = np.array(pred_sub)

pred_sub.shape
submission = pd.DataFrame()

submission['ImageId']=np.array([i+1 for i in range(len(pred_sub))])

submission['Label']=pred_sub

submission
submission.to_csv('submission.csv')