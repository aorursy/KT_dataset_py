%pylab inline



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler

from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedShuffleSplit

from keras.models import Sequential

from keras.layers import Merge

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers.advanced_activations import PReLU

from keras.layers import Convolution2D, MaxPooling2D

from keras.utils.np_utils import to_categorical

from keras.callbacks import EarlyStopping



## Read data from the CSV file

data = pd.read_csv('../input/train.csv')

parent_data = data.copy()    ## Always a good idea to keep a copy of original data

__id__ = data.pop('id')



data.shape

data.describe()



## Need to encode labels as they're strings

y = data.pop('species')

y = LabelEncoder().fit(y).transform(y)

print(y.shape)



## Normalizing data, zero mean

X = preprocessing.MinMaxScaler().fit(data).transform(data)

X = StandardScaler().fit(data).transform(data)

print(X.shape)

X



## We will be working with categorical crossentropy function

## It is required to further convert the labels into "one-hot" representation

y_cat = to_categorical(y)

print(y_cat.shape)



## retain class balances 

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1,random_state=12345)

train_id, value_id = next(iter(sss.split(X, y)))

x_train, x_val = X[train_id], X[value_id]

y_train, y_val = y_cat[train_id], y_cat[value_id]

print("x_train dim: ",x_train.shape)

print("x_val dim:   ",x_val.shape)

print()

# ----------------

## Developing a layered model for Neural Networks No/4/

## Input dimensions should be equal to the number of features

## We used softmax layer to predict a uniform probabilistic distribution of outcomes

Model = Sequential()

Model.add(Dense(900,input_dim=192,  init='uniform', activation='relu'))

Model.add(Dropout(0.25))

Model.add(Dense(450, activation='sigmoid'))

Model.add(Dropout(0.25))

Model.add(Dense(99, activation='softmax'))





Model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics = ["accuracy"])

early_stopping = EarlyStopping(monitor='val_loss', patience=600)

history = Model.fit(x_train, y_train,batch_size=192,nb_epoch=2500 ,verbose=0,validation_data=(x_val, y_val),callbacks=[early_stopping])

                    

print('val_acc: ',max(history.history['val_acc']))

print('val_loss: ',min(history.history['val_loss']))

print('train_acc: ',max(history.history['acc']))

print('train_loss: ',min(history.history['loss']))

print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))









## read test file

test = pd.read_csv('../input/test.csv')

index = test.pop('id')



## we need to perform the same transformations from the training set to the test set

test = preprocessing.MinMaxScaler().fit(test).transform(test)

test = StandardScaler().fit(test).transform(test)





yPred =  Model.predict_proba(test)



yPred = pd.DataFrame(yPred,index=index,columns=sort(parent_data.species.unique()))







fp = open('submission_nn_kernel.csv','w')

fp.write(yPred.to_csv())


