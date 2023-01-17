import os

import keras as kr

import pandas as pd

import numpy as np

import csv

from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

print(os.listdir("../input"))

##print(os.mkdir("../output"))

print(os.listdir(".."))
# fix random seed for reproducibility

seed = 42

np.random.seed(seed)
# load dataset

dataframe = pd.read_csv("../input/onehot/samples_oh_test.csv")

dataset = dataframe.values

X_test = dataset[:,:384].astype(int)

y_test = dataset[:,384:].astype(int)

print(X_test.shape)

print(y_test.shape)

dataframe = pd.read_csv("../input/onehot/samples_oh_train.csv")

dataset = dataframe.values

X_train = dataset[:,:384].astype(int)

y_train = dataset[:,384:].astype(int)

##print(dataset)

print(X_train.shape)

print(y_train.shape)
# encode class values as integers

##encoder = LabelEncoder()

##encoder.fit(y_train)

##encoded_Y = encoder.transform(y_train)

# convert integers to dummy variables (i.e. one hot encoded)

##dummy_y_train = np_utils.to_categorical(encoded_Y).astype(int)

##print(dummy_y_train)



##encoder.fit(y_test)

##encoded_Y = encoder.transform(y_test)

##dummy_y_test = np_utils.to_categorical(encoded_Y).astype(int)

# convert integers to dummy variables (i.e. one hot encoded)

##dummy_y_test = np_utils.to_categorical(encoded_Y).astype(int)

##print(dummy_y_test)
# create model

model = Sequential()

model.add(Dense(3000, input_dim=384, activation='relu'))

model.add(Dense(1800, input_dim=3000, activation='relu'))

model.add(Dense(600, input_dim=1800, activation='relu'))

model.add(Dense(300, input_dim=600, activation='relu'))

model.add(Dense(90, input_dim=300, activation='relu'))

model.add(Dense(3, activation='softmax'))

# Compile model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=10, epochs=500, verbose=1)

##results = cross_val_score(model, X_train, dummy_y_train, cv=kfold, scoring="accuracy" )

##print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
from sklearn.metrics import confusion_matrix

score = model.evaluate(X_test, y_test)

print('Score: ', score[1]*100)

y_pred = model.predict_classes(X_test, verbose=1)

print(y_pred)



encoder = LabelEncoder()

encoder.fit(y_pred)

encoded_Y = encoder.transform(y_pred)

dummy_y_test = np_utils.to_categorical(encoded_Y).astype(int)

from numpy import argmax

print(argmax(dummy_y_test))



conf_mx = confusion_matrix(argmax(y_test, axis=-1), y_pred)

print(conf_mx)

# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()
# serialize model to JSON

from keras.models import model_from_json

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")

print("Saved model to disk")