import numpy as np

import pandas as pd

import os

import cv2

import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import minmax_scale

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential

from keras.layers import Dense

from glob import glob

import os
# create dataset

train_data1 = glob("../input/fruits-360_dataset/fruits-360/Training/Raspberry/*")

train_data2 = glob("../input/fruits-360_dataset/fruits-360/Training/Pomelo Sweetie/*")



test_data1 = glob("../input/fruits-360_dataset/fruits-360/Test/Raspberry/*")

test_data2 = glob("../input/fruits-360_dataset/fruits-360/Test/Pomelo Sweetie/*")



x_train = []

x_test = []

def create_data(data, values):

    for i in values:

        im = cv2.imread(i, 0)

        data.append(im)

    return len(values)



# create train_data

train_Rasberry_size = create_data(x_train, train_data1)

train_Pomelo_size = create_data(x_train, train_data2)

print("train_Raspberry_size:{} || train_Pomelo_size:{}" .format(train_Rasberry_size, train_Pomelo_size))



# create test_data

test_Rasberry_size = create_data(x_test, test_data1)

test_Pomelo_size = create_data(x_test, test_data2)

print("test_Raspberry_size:{} || test_Pomelo_size:{}" .format(test_Rasberry_size, test_Pomelo_size))
# set train data

x_train = np.asarray(x_train, dtype=np.float64)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])



z = np.zeros(train_Rasberry_size)

o = np.ones(train_Pomelo_size)

y_train = np.concatenate((z,o), axis=0)



print("x_train:", x_train.shape)

print("y_train:", y_train.shape)
# set test data

x_test = np.asarray(x_test, dtype=np.float64)

x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])



z = np.zeros(test_Rasberry_size)

o = np.ones(test_Pomelo_size)

y_test = np.concatenate((z,o), axis=0)



print("x_test:", x_test.shape)

print("y_test:", y_test.shape)
# create ANN Model

def build_classifier():

    classifier = Sequential()

    classifier.add(Dense(units=16, kernel_initializer='uniform', activation='relu', input_dim=x_train.shape[1]))

    classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))

    classifier.add(Dense(units=8, kernel_initializer='uniform', activation='relu'))

    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classifier



model = KerasClassifier(build_fn=build_classifier, epochs=5, verbose=2)

results = cross_val_score(estimator=model, X=x_train, y=y_train, cv=3)

print("Accuracy mean:", results.mean())

print("Accuracy variance:", results.std())