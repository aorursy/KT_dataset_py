# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np


df = pd.read_csv('/kaggle/input/housepricesdata/housepricedata.csv')
df
dataset = df.values
dataset
X = dataset[:,0:10]
Y = dataset[:,10]
#Normalizing and Scaling the dataset are very important as there are no missing values in the dataset
#Normalization/Scaling makes input features to be on same order of magnitude
#Mim max scaler scales our data values to be between 0 and 1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale
#Data Splitting into Train, Validaion and Testing
from sklearn.model_selection import train_test_split
X_train, X_val_and_test, Y_train, Y_val_and_test = train_test_split(X_scale, Y, test_size=0.3)
X_val, X_test, Y_val, Y_test = train_test_split(X_val_and_test, Y_val_and_test, test_size=0.5)
#Getting the rows of all Train, Validation and testing
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)
#Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML. 
#Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible

#We will be using Keras to build Neural Network architecture using TensorFlow backend
!pip install Keras
!pip install tensorflow
#Importing required modules from Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
#Model 1: Sequential Model
#In Sequential Model, we need to describe the number of layers above in the sequence
# Here we used 2 Hidden layers with 30 neurons (ReLU activation), and 1 output layer with 1 neuron (Sigmoid function)

model = Sequential([
    Dense(32, activation='relu', input_shape=(10,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),
])

#Now, we got our architecture specified- we need to configure the model 
#optimizer=sgd (Stochastic gradient descent)
#loss=binary_crossentropy (giving the type of loss function to use)
#metrics=accuracy (Other metrics which you want to check apart from loss function)

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#Fitting the model
#epochs= tells how long we want to train the model

hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))
model.evaluate(X_test, Y_test)[1]
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
#Model 2: Training model with many hidden layers to check whether the model will overfit

model_2 = Sequential([
    Dense(1000, activation='relu', input_shape=(10,)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1, activation='sigmoid'),
])
model_2.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_2 = model_2.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))
plt.plot(hist_2.history['loss'])
plt.plot(hist_2.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()
plt.plot(hist_2.history['accuracy'])
plt.plot(hist_2.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()
#Now, you can see that Model 2 is over fitting when compared to Model 1 due to additional hidden layers added to Model 2 when compared to Model 1
#Model 3: Adding L2 regularization to same paramaters given to Model 2 thereby reducing the Overfitting in Model 2.
#L2 regularization also referred to as Ridge regression
#Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function
#L2 Regularization adds the term Lambda and Lambda need to be choosen carefully.
#L2 regularization forces the weights to be small but does not make them zero and does non sparse solution.

from keras.layers import Dropout
from keras import regularizers

model_3 = Sequential([
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(10,)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1000, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    Dropout(0.3),
    Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)),
])


model_3.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
hist_3 = model_3.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_val, Y_val))
plt.plot(hist_3.history['loss'])
plt.plot(hist_3.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.ylim(top=1.2, bottom=0)
plt.show()
#As you can see that the Model 3 (With Regularization) is less overfitting when compared to Model 2 with many hidden layers (no regularization)
#Regularization improves the performance of the models
plt.plot(hist_3.history['accuracy'])
plt.plot(hist_3.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()