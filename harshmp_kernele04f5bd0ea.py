# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,ZeroPadding2D,BatchNormalization,Activation,Input

from keras.optimizers import RMSprop

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau

from keras.callbacks import ModelCheckpoint



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
Y_train = train['label']

X_train = train.drop('label',axis = 1) 
X_train.isnull().any().describe()
test.isnull().any().describe()
X_train = X_train/255

test = test/255
X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)

Y_train  = to_categorical(Y_train,num_classes = 10)
def model(input_shape):

    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input)

    # conv>batchnorm>relu

    X = Conv2D(32,(7,7),strides = (1,1),name = 'conv0')(X)

    X = BatchNormalization(axis = 3,name= 'bn0')(X)

    X = Activation('relu')(X)

    X = MaxPool2D((2,2),name = 'maxpool0')(X)

    X = Conv2D(32,(7,7),strides = (1,1),name = 'conv1')(X)

    X = BatchNormalization(axis = 3,name= 'bn1')(X)

    X = Activation('relu')(X)

    X = MaxPool2D((2,2),name = 'maxpool1')(X)

    X = Flatten()(X)

    X = Dense(80,activation= 'relu',name = 'fc0')(X)

    X = Dense(40,activation= 'relu',name = 'fc1')(X)

    X = Dense(10,activation= 'sigmoid',name = 'fc2')(X)

    model = Model(inputs = X_input,outputs = X,name = 'First')

    return model
First_attempt = model((28,28,1))
First_attempt.compile(optimizer = 'RMSprop',loss = 'categorical_crossentropy',metrics = ['accuracy'])
filepath = "weights.best_csv"

checkpoint = ModelCheckpoint(filepath,monitor = "val_acc",save_best_only = True, mode='max')

callbacks_list = [checkpoint]

First_attempt.fit(x = X_train,y = Y_train,validation_split = 0.33,epochs = 5,batch_size = 112,callbacks = callbacks_list,verbose = 0)
result = First_attempt.predict(test)
result = np.argmax(result,axis = 1)
result = pd.Series(result,name = 'Label')
submission = pd.concat([pd.Series(range(1,28001),name = 'ImageId'),result],axis = 1)

submission.to_csv("cnn_result")