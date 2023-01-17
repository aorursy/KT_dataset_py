import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import tensorflow as tf

from six.moves import range

from sklearn.model_selection import train_test_split

import os

import sys



from IPython.display import display, Image

import matplotlib.pyplot as plt



# Importing the dataset

dataset = pd.read_csv('../input/diabetes.csv')



# split into input (X) and output (Y) variables

X = dataset.drop(['Outcome'], axis=1)

Y = dataset['Outcome']



# split into 67% for train and 33% for test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)



import keras

from keras.models import Sequential

from keras.layers import Dense



# create model

model = Sequential()

model.add(Dense(12, input_dim=8, init= 'uniform' , activation= 'relu' ))

model.add(Dense(8, init= 'uniform' , activation= 'relu' ))

model.add(Dense(1, init= 'uniform' , activation= 'sigmoid' ))



# Compile model

model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])



# Fit the model

model.fit(X_train, y_train, validation_data=(X_test,y_test), nb_epoch=150, batch_size=10)



# evaluate the model

scores = model.evaluate(X, Y)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


