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
Train = pd.read_csv("../input/mnist_train.csv")

Test = pd.read_csv("../input/mnist_test.csv")
Train.shape, Test.shape
img = np.array(Train.iloc[0,1:]) #for row 0
img.shape
img = img.reshape(28,28) #reshaping to 2d shape from linear vector

img.shape
import matplotlib.pyplot as plt

#plt.imshow(img, cmap="gray") #Grayscale image

plt.imshow(img)

plt.show()
X_train = np.array(Train.iloc[:, 1:])

y_train = np.array(Train.iloc[:, 0])

X_test = np.array(Test.iloc[:, 1:])

y_test = np.array(Test.iloc[:, 0])
X_train.shape, y_train.shape, X_test.shape, y_test.shape

#Keras 3 steps

#1.Structure, 2.Build, 3.Fit



from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train, 10) #to_categorical implements One Hot Encoding

y_test = np_utils.to_categorical(y_test, 10)
y_train[0]
import keras

from keras.models import Sequential #Sequential for connected neural networks like ANN, CNN

from keras.layers import Dense, Activation #Dense -> creates Dense layer for ANN, Activation-> Activation function like Sigmoid
model = Sequential()

model.add(Dense(1024, input_shape=(784,))) #512->Output layer, 784->Input layer. 512 neurons go into next layer

model.add(Activation('relu')) #Normalizes using relu

model.add(Dense(10)) #10 neurons in Dense layer (Neurons in Output layer i.e No of classes)

model.add(Activation('softmax')) #Softmax for multiple class classification, Sigmoid for Binary Class Classification



#Tuning is the process for changing no of neurons in Hidden layers

#Weight and Bias are parameters. It's tuning is done automatically using Gradient Descend

model.summary()
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')

#Categorical for multiple classification

#Binary for binary classification
#fitting model

model.fit(X_train, y_train, epochs=10)
model.evaluate(X_test, y_test)