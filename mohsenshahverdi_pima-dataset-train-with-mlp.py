# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import theano

from keras.models import Sequential

from keras.layers import Dense

from theano import tensor

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



seed = 7

# fix random seed for reproducibility

np.random.seed(seed)

# load pima indians dataset

dataset = pd.read_csv("../input/pima-indians-diabetes-database/diabetes.csv")

# split into input (X) and output (Y) variables

X = dataset.iloc[:,0:8].values

Y = dataset.iloc[:,8].values



# split into 67% for train and 33% for test

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)



# create model

model = Sequential()

model.add(Dense(32, input_dim=8, activation='relu'))

model.add(Dense(24, activation='relu'))

model.add(Dense(12, activation='relu'))

model.add(Dense(6,  activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model

history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=400, batch_size=40)



# evaluate the model

scores = model.evaluate(X, Y)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

import matplotlib.pyplot as plt

# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()