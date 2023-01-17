

import numpy as np

import tensorflow as tf

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from numpy import array

from numpy import argmax

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import seaborn as sns

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Importing Libraries and display the records

import numpy as np

import pandas as pd

from time import time

from IPython.display import display

from sklearn.metrics import accuracy_score

from sklearn.datasets import load_digits

#from sklear



# Pretty display for notebooks

%matplotlib inline



# Load the Census da/kaggle/input/hr-attrition/HR_comma_sep.csvtaset

data = pd.read_csv("/kaggle/input/hr-attrition/HR_comma_sep.csv")

#data=load_digits()



# Success - Display the first record

display(data.head(5))
# learning rate and initial lambda initialization

lr = .0001

init_lam = .01



# modify the epochNum, or number of iterations that our network runs

epochNum = 2
y = data['left']

X = data.drop('left', axis = 1)

# One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()

X = pd.get_dummies(X)



# Encode the 'left_raw' data to numerical values

X = pd.get_dummies(X, drop_first = True)
# Use Sci-Kit's one-hot encoder to transform the classes into numerical values

# to feed into our multilayer perceptron

numVal = array(y)



# integer encode

label_encoder = LabelEncoder()

integer_encoded = label_encoder.fit_transform(numVal)



# encode from integer to binary

onehot_encoder = OneHotEncoder(sparse=False)

integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

onehot_encoded = onehot_encoder.fit_transform(integer_encoded)



# splitting up training and testing data into appropriate vectors for initial 

# group

X_train, X_test, y_train, y_test = train_test_split(X, onehot_encoded)



# establish the number of labels and features

num_labels = y_train.shape[1]

num_features = X_train.shape[1]



# set size of hidden layer

hidden_nodes = 3 * len(X_train)



# establish the arrays of weights and biases

w1 = np.random.normal(0, 1, [num_features, hidden_nodes]) 

w2 = np.random.normal(0, 1, [hidden_nodes, num_labels]) 



b1 = np.zeros((1, hidden_nodes))

b2 = np.zeros((1, num_labels))
def relu_activation(vec):

  return np.maximum(vec, 0)





# returns a vector of output probabilities

def softmax(vec):

  # for softmax we compute input over number of choices

  input = np.exp(vec)

  # output is sum of all of those choices, K

  output = np.sum(input, axis = 1, keepdims = True)

  return input / output





def forward(softmax_vec, onehot_labels, lam, w1, w2):

  

  # first we calculate softmax cross-entropy loss (refer to formula)

  i = np.argmax(onehot_labels, axis = 1).astype(int)

  

  # since softmax output will be probability values (non-integer) we use function

  # arange() 

  predicted = softmax_vec[np.arange(len(softmax_vec)), i]

  logs = np.log(predicted)

  loss = -np.sum(logs) / len(logs)

  

  # second we add regularization to the loss in order to avoid overfitting

  w1_loss = 0.5 * lam * np.sum(w1 * w1)

  w2_loss = 0.5 * lam * np.sum(w2 * w2)

  return (loss + (w1_loss + w2_loss))

  

  

def backprop(w1, b1, w2, b2, lam, lr, output_vec, hidden_vec):

  output_error = (output_vec - y_train) / output_vec.shape[0]



  hidden_error = np.dot(output_error, w2.T) 

  hidden_error[hidden_vec <= 0] = 0



  gw2 = np.dot(hidden_vec.T, output_error)

  gb2 = np.sum(output_error, axis = 0, keepdims = True)



  gw1 = np.dot(X_train.T, hidden_error)

  gb1 = np.sum(hidden_error, axis = 0, keepdims = True)



  gw2 += lam * w2

  gw1 += lam * w1



  w1 -= lr * gw1

  b1 -= lr * gb1

  w2 -= lr * gw2

  b2 -= lr * gb2
# since we need to return the object 'epoch' in this case we will use xrange()

# rather than range() function in python



for epoch in range(1,epochNum):

  # wx + b

  input = np.dot(X_train, w1) + b1

  hidden = relu_activation(input)

  output = np.dot(hidden, w2) + b2

  soft_output = softmax(output)



  forward(soft_output, y_train, init_lam, w1, w2)

  backprop(w1, b1, w2, b2, init_lam, lr, output, hidden)
# test



def eval(preds, y):

    ifcorrect =  np.argmax(preds, 1) == np.argmax(y, 1)

    correct_predictions = np.sum(ifcorrect)

    return correct_predictions * 100 / preds.shape[0]

  



input = np.dot(X_test, w1)

hidden = relu_activation(input + b1)

scores = np.dot(hidden, w2) + b2

probs = softmax(scores)

print('Accuracy of Multilayer Perceptron: {0}%'.format(eval(probs, y_test)))