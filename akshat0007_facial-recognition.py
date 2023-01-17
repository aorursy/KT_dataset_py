# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/facial-expression/fer2013/"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf



import keras

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D

from keras.layers import Dense, Activation, Dropout, Flatten



from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# get the data

filname = '../input/facial-expression/fer2013/fer2013.csv'

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

names=['emotion','pixels','usage']

df=pd.read_csv('../input/facial-expression/fer2013/fer2013.csv',names=names, na_filter=False)

im=df['pixels']

df.head(10)
def getData(filname):

    # images are 48x48

    # N = 35887

    Y = []

    X = []

    first = True

    for line in open(filname):

        if first:

            first = False

        else:

            row = line.split(',')

            Y.append(int(row[0]))

            X.append([int(p) for p in row[1].split()])



    X, Y = np.array(X) / 255.0, np.array(Y)

    return X, Y

X, Y = getData(filname)

num_class = len(set(Y))

print(num_class)
# keras with tensorflow backend

N, D = X.shape

X = X.reshape(N, 48, 48, 1)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=10)

y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)

y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
from keras.models import Sequential

from keras.layers import Dense , Activation , Dropout ,Flatten

from keras.layers.convolutional import Conv2D

from keras.layers.convolutional import MaxPooling2D

from keras.metrics import categorical_accuracy

from keras.models import model_from_json

from keras.callbacks import ModelCheckpoint

from keras.optimizers import *

from keras.layers.normalization import BatchNormalization
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, load_model

from keras.layers import Activation, Dropout, Flatten, Dense

from keras.applications.inception_resnet_v2 import InceptionResNetV2

from keras.applications.inception_v3 import InceptionV3

from keras.callbacks import ModelCheckpoint

from keras import metrics

from keras.optimizers import Adam 

from keras import backend as K

import keras
def my_model():

    model = Sequential()

    input_shape = (48,48,1)

    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))

    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))



    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))

    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    

    model.add(Flatten())

    model.add(Dense(128))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(Dropout(0.2))

    model.add(Dense(7))

    model.add(Activation('softmax'))

    

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

    # UNCOMMENT THIS TO VIEW THE ARCHITECTURE

    #model.summary()

    

    return model

model=my_model()

model.summary()
path_model='model_filter.h5' # save model at this location after each epoch

K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one

model=my_model() # create the model

K.set_value(model.optimizer.lr,0.0005) # set the learning rate

# fit the model

h=model.fit(x=X_train,     

            y=y_train, 

            batch_size=64, 

            epochs=20, 

            verbose=1, 

            validation_data=(X_test,y_test),

            shuffle=True,

            callbacks=[

                ModelCheckpoint(filepath=path_model),

            ]

            )