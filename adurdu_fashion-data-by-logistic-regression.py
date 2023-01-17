# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # visualization tool

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data_train = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashionmnist/fashion-mnist_test.csv')
X_train = data_train.iloc[:,1:785].values

X_test = data_test.iloc[:,1:785].values
print('X_train', X_train.shape)

print('X_test', X_test.shape)
y_train = data_train.iloc[:,0].values

y_test = data_test.iloc[:,0].values
y_train = to_categorical(y_train)

y_test = to_categorical(y_test)
print('y_train', y_train.shape)

print('y_test', y_test.shape)
x_train = X_train.T

x_test = X_test.T

Y_train = y_train.T

Y_test = y_test.T

print("x train: ",x_train.shape)

print("x test: ",x_test.shape)

print("y train: ",Y_train.shape)

print("y test: ",Y_test.shape)
#normalization 

x_train = (x_train-np.min(x_train))/(np.max(x_train)-np.min(x_train))

x_test = (x_test-np.min(x_test))/(np.max(x_test)-np.min(x_test))
#print(Y_train[:,:])
# short description and example of definition (def)

def dummy(parameter):

    dummy_parameter = parameter + 5

    return dummy_parameter

result = dummy(3)     # result = 8



# lets initialize parameters

# So what we need is dimension 4096 that is number of pixels as a parameter for our initialize method(def)

def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w, b
# calculation of z

#z = np.dot(w.T,x_train)+b

def sigmoid(z):

    y_head = 1/(1+np.exp(-z))

    return y_head
y_head = sigmoid(0)

y_head
# Forward propagation steps:

# find z = w.T*x+b

# y_head = sigmoid(z)

# loss(error) = loss(y,y_head)

# cost = sum(loss)

def forward_propagation(w,b,x_train,y_train):

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z) # probabilistic 0-1

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    return cost 



# In backward propagation we will use y_head that found in forward progation

# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
# Updating(learning) parameters

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list

#parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate = 0.009,number_of_iterarion = 200)



 # prediction

def predict(w,b,x_test):

    # x_test is a input for forward propagation

    z = sigmoid(np.dot(w.T,x_test)+b)

    Y_prediction = np.zeros((1,x_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(z.shape[1]):

        if z[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
def logistic_regression(x_train, Y_train, x_test, Y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, Y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - Y_test)) * 100))

    

logistic_regression(x_train, Y_train, x_test, Y_test,learning_rate = 0.0005, num_iterations = 300)