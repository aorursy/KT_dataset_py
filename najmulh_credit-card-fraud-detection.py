# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read data from file

data_file_path = "../input/creditcard.csv"

creditcard_data = pd.read_csv(data_file_path)

# Describe

creditcard_data.head()

from sklearn.utils import shuffle



# Features

# Objective is to have equal propotion of fraud and normal transactions in both train and test data

# Reference https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow



cc_normal = creditcard_data[creditcard_data.Class==0]

cc_fraud = creditcard_data[creditcard_data.Class==1]



x_train = pd.concat([cc_normal.sample(frac=0.8), cc_fraud.sample(frac=0.8)], axis=0)

x_train = shuffle(x_train)



y_train = x_train.Class

x_train = x_train.drop(['Class'], axis = 1)



x_test = pd.concat([cc_normal.sample(frac=0.2), cc_fraud.sample(frac=0.2)], axis=0)

x_test = shuffle(x_test)



y_test = x_test.Class

x_test = x_test.drop(['Class'], axis = 1)



# Input shape

x_train.shape
# Fully connected neural network

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras.optimizers import RMSprop



model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(64,input_shape=(30,)),

  tf.keras.layers.Dense(512, activation='relu'),

  tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=1e-4),

              metrics=['acc'])



model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5)
# Something fishy!!

# Classes are skewed. Let's try class_weight

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',

                                                 np.unique(y_train),

                                                 y_train)

print(class_weights)

model.fit(x_train, y_train, validation_data=(x_test, y_test), class_weight=class_weights, epochs=5)
# Let's remove amount and run the same model



x_train = x_train.drop(['Amount'], axis = 1)

x_test = x_test.drop(['Amount'], axis = 1)



model = tf.keras.models.Sequential([

  tf.keras.layers.Dense(64,input_shape=(29,)),

  tf.keras.layers.Dense(512, activation='relu'),

  tf.keras.layers.Dense(1, activation='sigmoid')

])





model.compile(loss='binary_crossentropy',

              optimizer=RMSprop(lr=1e-4),

              metrics=['acc'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), class_weight=class_weights, epochs=2)
# PLOT LOSS AND ACCURACY

%matplotlib inline



import matplotlib.image  as mpimg

import matplotlib.pyplot as plt



#-----------------------------------------------------------

# Retrieve a list of list results on training and test data

# sets for each training epoch

#-----------------------------------------------------------

acc=history.history['acc']

val_acc=history.history['val_acc']

loss=history.history['loss']

val_loss=history.history['val_loss']



epochs=range(len(acc)) # Get number of epochs



#------------------------------------------------

# Plot training and validation accuracy per epoch

#------------------------------------------------

plt.plot(epochs, acc, 'r', "Training Accuracy")

plt.plot(epochs, val_acc, 'b', "Validation Accuracy")

plt.title('Training and validation accuracy')

plt.figure()



#------------------------------------------------

# Plot training and validation loss per epoch

#------------------------------------------------

plt.plot(epochs, loss, 'r', "Training Loss")

plt.plot(epochs, val_loss, 'b', "Validation Loss")





plt.title('Training and validation loss')