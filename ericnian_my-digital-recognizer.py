# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import keras

from keras.datasets import mnist

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten

#from keras.layers import Conv2D, MaxPooling2D

from keras import backend as K

from sklearn.model_selection import train_test_split



from keras.callbacks import ModelCheckpoint

from keras.callbacks import EarlyStopping

from keras import models



import seaborn as sns

import matplotlib.pyplot as plt

import os,sys



os.getcwd()
#load data

import csv



#load data to pd DataFrame

trainSet = pd.read_csv('../input/digit-recognizer/train.csv')

testSet = pd.read_csv('../input/digit-recognizer/test.csv')

#show data

print("shape of trainSet : ",trainSet.shape)

print("shape of testSet : ",testSet.shape)

numberOfTest = len(testSet)

m = len(trainSet)

print("numbers of samples :",len(trainSet))

print("keys of trainSet :",trainSet.columns)

#drop index from trainSet

X_train = trainSet.drop("label",axis = 1)

print("shape of X_train :",X_train.shape)

Y_train = trainSet["label"]

print("shape of Y_train is :",Y_train.shape)

#print("type of Y_train is : ",Y_train.type)

#numberOfY_train = trainSet["label"].shape

#Y_train = np.array(Y_train)

#Y_train = Y_train.reshape((numberOfY_train,1))

#print("shape of Y_train :",Y_train.shape)



#split train set to train and vailid set

X_trainOnly = X_train[:m - numberOfTest]

X_trainForValid = X_train[m - numberOfTest:]

(X_trainOnly.shape,X_trainForValid.shape)



Y_trainOnly = Y_train[:m - numberOfTest]

Y_trainForValid = Y_train[m - numberOfTest:]

print("valid shape is ",Y_trainForValid.shape,X_trainForValid.shape)

X_trainOnly.head()

print("Y_train shape is ",Y_train.index)
#standized the data

X_trainOnly = X_trainOnly.astype("float32")

X_trainForValid = X_trainForValid.astype("float32")

testSet = testSet.astype("float32")

X_trainOnly /= 255

X_trainForValid /= 255

testSet /= 255



print("shape of X_trainOnly is : ",X_trainOnly.shape)

print(Y_train.value_counts())

sns.countplot(Y_trainOnly)

print("shape of X_valid and Y_valid :"

      ,X_trainForValid.shape,Y_trainForValid.shape)





X_trainOnly.shape,Y_trainOnly.shape,X_trainForValid.shape,Y_trainForValid.shape


# conversion of label to binary form

numberOfClasses = 10

train_label = keras.utils.to_categorical(Y_trainOnly,numberOfClasses)

valid_label = keras.utils.to_categorical(Y_trainForValid,numberOfClasses)



train_label.shape,valid_label.shape

#build model

model = Sequential()

model.add(Dense(256,input_dim = 784,activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(128,activation = 'relu'))

model.add(Dropout(0.25))

model.add(Dense(64,activation = 'relu'))

model.add(Dense(numberOfClasses,activation = 'softmax'))
model.summary()
model.compile(loss = keras.losses.categorical_crossentropy,

             optimizer = keras.optimizers.Adam(),

             metrics = ['accuracy'])

es = EarlyStopping(monitor='accuracy', patience=5)

md = ModelCheckpoint(filepath="/content/best_model.h5", verbose=1, save_best_only=True)
history = model.fit(X_trainOnly,train_label,

                    batch_size = 128,

                    epochs = 10,

                    verbose = 1,

                    validation_data = (X_trainForValid,valid_label))
