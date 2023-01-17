# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras as kr # To build Neural net
import scipy as sp # linear algebra
import matplotlib as mpts #to visulaize the data

import pylab as pl #to visulaize the data

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#to have the matplotlib graphs included in notebook
%matplotlib inline
#from tensorflow.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets("MNIST_data/")
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

train_data.shape
train_data['label']
#get the only pixel data and drop label
train_inputs =train_data.drop(['label'],axis=1)
#get labels in diff datframe
train_labels=train_data[:]['label']
train_inputs.shape
train_labels.shape
#Random forest from scikit
from sklearn import ensemble
classifier=ensemble.RandomForestClassifier()
classifier.fit(train_inputs,train_labels)
#Accuracy with the trained data
score=classifier.score(train_inputs,train_labels)
print(score)
#Testing with the test data
i=15809
print(classifier.predict(test_data[i:i+1])[0])
import matplotlib.pyplot as plt
plt.imshow(test_data[i:i+1].values.reshape(28,28))
#training with neural net using keras
import sklearn as sk #Importing scikit
from keras.models import Sequential#importing Sequential to build neural net
##adding layers to neural network
model = Sequential()
from keras.layers import Dense

model.add(Dense(units=784, activation='relu', input_dim=784))
model.add(Dense(units=100, activation='relu', input_dim=784))
model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
#CHnaging the target data to make use in a neural network final layer
from keras.utils.np_utils import to_categorical
categorical_results = to_categorical(train_labels, num_classes=None)
#Train the neural net with 10 epochs we are trainng with less data only
model.fit(train_inputs, categorical_results, epochs=10, batch_size=784)
#Test the random image from the dataset
i=22507
a=test_data[i:i+1].values
b=model.predict(a)
pl.imshow(a.reshape(28,28))
#mapping = pd.read_csv('C:/Users/bnaveen3/Desktop/mapping.csv',header=None)
#
#results.shape
#print(b)
np.max(b)
print ('Predicted value is',np.where(b == np.max(b))[1][0])
