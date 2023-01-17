# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

import os

from keras.preprocessing.image import img_to_array, load_img

from keras.utils import np_utils



from keras.applications.vgg16 import VGG16

from keras.applications.inception_v3 import InceptionV3

from keras.models import Sequential

from keras.layers import Convolution2D,MaxPooling2D,Dropout,Dense,Flatten,BatchNormalization,Conv2D

from sklearn.model_selection import train_test_split



from keras.preprocessing.image import ImageDataGenerator

from keras.optimizers import RMSprop

from keras import backend as K
df_train = pd.read_csv('../input/digit-recognizer/train.csv')

df_test = pd.read_csv('../input/digit-recognizer/test.csv')
print(df_train.head())

print(df_test.head())
X_train = df_train.drop(['label'], axis=1)

y_train = df_train['label']

X_train = np.array(X_train)

X_train = X_train/255

X_train = X_train.reshape(-1,28,28,1)

y_train = np_utils.to_categorical(y_train, num_classes = 10)
print(X_train.shape)

print(y_train.shape)
X_test = df_test

X_test = np.array(X_test)

X_test = X_test/255

X_test = X_test.reshape(-1,28,28,1)
del df_train,df_test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=2)
def Create_Model_CNNbuild(height, width, classes, channels):

    model = Sequential()

    

    inputShape = (height, width, channels)





    model.add(Conv2D(32, (5,5),padding='same', activation = 'relu', input_shape = inputShape))

    model.add(Conv2D(32, (5,5),padding='same', activation = 'relu', input_shape = inputShape))

    #model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(2,2))

    model.add(Dropout(0.25))



    model.add(Conv2D(64, (3,3),padding='same', activation = 'relu'))

    model.add(Conv2D(64, (3,3),padding='same', activation = 'relu'))

    #model.add(BatchNormalization(axis = -1))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))





    model.add(Flatten())

    model.add(Dense(256, activation = 'relu'))

    #model.add(BatchNormalization(axis = -1))

    model.add(Dropout(0.5))

    model.add(Dense(classes, activation = 'softmax'))

    

    return model
model = Create_Model_CNNbuild(height=28, width=28, channels=1, classes=10)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5,min_lr=0.00001)
datagen = ImageDataGenerator(rotation_range=10, zoom_range = 0.1, width_shift_range=0.1, height_shift_range=0.1)  

datagen.fit(X_train)

data_from_generator = datagen.flow(X_train,y_train, batch_size = 86)
model_fit_history = model.fit_generator(data_from_generator,epochs = 30, validation_data = (X_val,y_val),verbose = 1, steps_per_epoch = X_train.shape[0] // 86, callbacks=[learning_rate_reduction])
y_hat = model.predict_classes(X_test)
y_hat.shape
np.savetxt('MNIST_results.csv',y_hat ,delimiter=',')