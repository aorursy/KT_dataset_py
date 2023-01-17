# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

import scipy.stats   

from sklearn import preprocessing

from keras import models

from keras import layers



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
wine = pd.read_csv('../input/winequality.csv', delimiter =";")
wine.info()
wine.describe()
wine.head()
wine.groupby ('iswhite')['iswhite'].count()
wine["alcohol"]=wine["alcohol"].astype(int)
#Scaling the continuos variables

wine_scale = wine.copy()

scaler = preprocessing.StandardScaler()

columns =wine.columns[0:13]

wine_scale[columns] = scaler.fit_transform(wine_scale[columns])

wine_scale.head()

wine_scale = wine_scale.iloc[:,0:13]



wine_scale.describe()
sample = np.random.choice(wine_scale.index, size=int(len(wine_scale)*0.8), replace=False)

train_data, test_data = wine_scale.iloc[sample], wine_scale.drop(sample)



print("Number of training samples is", len(train_data))

print("Number of testing samples is", len(test_data))

print(train_data[:10])

print(test_data[:10])
train_data
features = train_data.drop('iswhite', axis=1)

targets = train_data['iswhite']

targets = targets > 0.5

features_test = test_data.drop('iswhite', axis=1)

targets_test = test_data['iswhite']

targets_test = targets_test > 0.5
def sigmoid(x):

    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):

    return sigmoid(x) * (1-sigmoid(x))

def error_formula(y, output):

    return - y*np.log(output) - (1 - y) * np.log(1-output)

def error_term_formula(y, output):

    return (y-output) * output * (1 - output)
epochs = 9000

learnrate = 0.3



# Training function

def train_nn(features, targets, epochs, learnrate):

    

    # Use to same seed to make debugging easier

    np.random.seed(42)



    n_records, n_features = features.shape

    last_loss = None



    # Initialize weights

    weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

    print(weights.shape)



    for e in range(epochs):

        del_w = np.zeros(weights.shape)

        for x, y in zip(features.values, targets):

            # Loop through all records, x is the input, y is the target



            # Activation of the output unit

            #   Notice we multiply the inputs and the weights here 

            #   rather than storing h as a separate variable 

            output = sigmoid(np.dot(x, weights))



            # The error, the target minus the network output

            error = error_formula(y, output)



            # The error term

            #   Notice we calulate f'(h) here instead of defining a separate

            #   sigmoid_prime function. This just makes it faster because we

            #   can re-use the result of the sigmoid function stored in

            #   the output variable

            error_term = error_term_formula(y, output)



            # The gradient descent step, the error times the gradient times the inputs

            del_w += error_term * x



        # Update the weights here. The learning rate times the 

        # change in weights, divided by the number of records to average

        weights += learnrate * del_w / n_records



        # Printing out the mean square error on the training set

        if e % (epochs / 10) == 0:

            out = sigmoid(np.dot(features, weights))

            loss = np.mean((out - targets) ** 2)

            print("Epoch:", e)

            if last_loss and last_loss < loss:

                print("Train loss: ", loss, "  WARNING - Loss Increasing")

            else:

                print("Train loss: ", loss)

            last_loss = loss

            print("=========")

    print("Finished training!")

    return weights

    

weights = train_nn(features, targets, epochs, learnrate)
tes_out = sigmoid(np.dot(features_test, weights))

predictions = tes_out > 0.5

accuracy = np.mean((predictions == targets_test))

print("Prediction accuracy: {:.3f}".format(accuracy))
features = train_data.drop('iswhite', axis=1)

targets = train_data['iswhite']

features_test = test_data.drop('iswhite', axis=1)

targets_test = test_data['iswhite']
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.optimizers import SGD

from keras.utils import np_utils



# Building the model

model = Sequential()

model.add(Dense(1, activation='softmax', input_shape=(12,)))

model.add(Dense(1, activation='softmax'))



# Compiling the model

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
model.fit(features, targets, epochs=9000, batch_size=100, verbose=0)