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
# Import modules



import numpy as np

import pandas as pd

import keras

import tensorflow as tf

from matplotlib import pyplot as plt 

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
# Download dataset



data = pd.read_csv('../input/mushrooms.csv')

data.head()
# Preparation of training data for the neural network

# The variable x_data includes all columns except the class column; the drop function removes the class column

# The variable x_data includes the class column



x_data = data.drop(['class'], axis=1)

y_data = data['class']
# Converting categorical data into numeric, because you can not use text in the data to train the model

# To convert text to numeric data, use the LabelEncoder class



labelencoder_x = LabelEncoder()

for col in x_data.columns:

    x_data[col] = labelencoder_x.fit_transform(x_data[col])



labelencoder_y=LabelEncoder()

y_data = labelencoder_y.fit_transform(y_data)
x_data.head()
print(y_data)
print(len(y_data), len(x_data))
# Randomize data division into training data and test data using train_test_split,

# as the 1 and 2 parameters we transfer data (x_data, y_data), test_size = 0.25 - the parameter indicates how much

# of data will be included in the test suite, in this case 25%.



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=42)
print(len(x_train), len(y_train))

print(len(x_test), len(y_test))
# to_categorical converts the vector y_train into a matrix (the number of columns is equal to the number of classes,

# number of lines remains the same), consisting of zeros and ones.



y_test_one_hot = keras.utils.to_categorical(y_test)

y_train_one_hot = keras.utils.to_categorical(y_train)



print(y_train_one_hot.shape)
# Normalization / Standardization of data

# StandardScaler () normalizes the data, so each column will have mean = 0 and standard deviation = 1

# Usually the data set contains variables of different scale (the values of the columns in x_train have different values, from 0 to 11)

# Since the values in the columns differ in scale, they are normalized so that they have a common scale, and

# After applying StandardScaler (), each column in x_train will have an average value of 0 and a standard deviation of 1



standard_scaler = StandardScaler()



x_train = standard_scaler.fit_transform(x_train)

x_test = standard_scaler.transform(x_test)
print(x_train)
# Build a model. The model consists of an input layer, 2 hidden layers (in 1 hidden layer there are 100 neurons, the activation function is relu,

# in 2 hidden layers of 50 neurons, the activation function is relu) and the output layer, which consists of 2 neurons,

# each of which represents a particular class, poisonous (eatable) and eatable (edible), the activation function is softmax





model = tf.keras.models.Sequential()



model.add(tf.keras.layers.BatchNormalization(input_shape=x_train.shape[1:]))



model.add(tf.keras.layers.Dense(100))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(50))

model.add(tf.keras.layers.Activation('relu'))

model.add(tf.keras.layers.Dense(2))

model.add(tf.keras.layers.Activation('softmax'))



model.build()

model.summary()
# Compile the model. Definition of optimizer (optimization function), loss (loss function), metrics (metrics)



model.compile(optimizer=tf.train.AdamOptimizer(), 

              loss='categorical_crossentropy',

              metrics=['accuracy'])
# Teaching model



model.fit(x_train, y_train_one_hot, epochs=20, batch_size=100)
# Use the trained model to get predictions on test data



predictions = model.predict_classes(x_test, verbose=0)
# Check what accuracy the model gives on test data



test_loss, test_accuracy = model.evaluate(x_test, y_test_one_hot)

print("Accuracy on test dataset: ", test_accuracy)