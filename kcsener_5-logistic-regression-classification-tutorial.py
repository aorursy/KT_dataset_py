# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#load dataset

#X is image array, y is label array

X = np.load('../input/Sign-language-digits-dataset/X.npy')

y = np.load('../input/Sign-language-digits-dataset/Y.npy')
#visual representation of one of 0 and 1

img_size = 64

plt.subplot(1,2,1)

plt.imshow(X[260].reshape(img_size, img_size))

plt.axis('off')

plt.subplot(1,2,2)

plt.imshow(X[900].reshape(img_size, img_size))

plt.axis('off')

plt.show()
#We use 0 and 1s; so create arrays for this:

#from 204 to 409 are zero sign, 822 to 1027 are one sign

X = np.concatenate((X[204:409], X[822:1027]), axis = 0)



print(X.shape)



zeros = np.zeros(205)

ones = np.ones(205)

y = np.concatenate((zeros, ones), axis=0).reshape(X.shape[0], 1)

print(y.shape)
#In order to use as input, we need to flatten the shape of X:

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

print(X.shape)



#410,4096 means that X data set have 410 samples and 4096 features
#Import function

from sklearn.model_selection import train_test_split



#Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train = X_train.T

X_test = X_test.T

y_train = y_train.T

y_test = y_test.T



print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#We need dimension which is number of pixels 4096 as parameter

def initialize_weights_and_bias(dimension):

    w = np.full((dimension,1), 0.01) #full method 4096'ya 1'lik bir matrix oluşturuyor, bu matrix'in içine de 0.01'i koyar.

    b = 0.0

    return w, b #w= 4096,1'lik ve 0.01'den oluşan bir matrix
#z = = b + px1*w1 + px2*w2 + ... + px4096*w4096 = np.dot(w.T, X_train)+b



def sigmoid(z):

    y_head = 1 / (1+ np.exp(-z)) #sigmoid function

    return y_head  #y_head = sigmoid(z)
#find z = w.T * X +b

#y_head= sigmoid(z)

#loss(error) = loss(y, y_head)

#cost = sum(loss)



def forward_propagation(w,b,X_train, y_train):

    z = np.dot(w.T, X_train)+b

    y_head = sigmoid(z) #0-1 arası bir değer verdi

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/X_train.shape[1]      # x_train.shape[1]  is for scaling

    return cost
# In backward propagation we will use y_head that found in forward progation

# Therefore instead of writing backward propagation method, lets combine forward propagation and backward propagation

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/X_train.shape[1]      # x_train.shape[1]  is for scaling

    # backward propagation

    derivative_weight = (np.dot(X_train,((y_head-y_train).T)))/X_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/X_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}

    return cost,gradients
# Updating(learning) parameters

def update(w, b, X_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,X_train,y_train)

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

def predict(w,b,X_test):

    # x_test is a input for forward propagation

    y_head = sigmoid(np.dot(w.T,X_test)+b)

    Y_prediction = np.zeros((1,X_test.shape[1]))

    # if z is bigger than 0.5, our prediction is sign one (y_head=1),

    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),

    for i in range(y_head.shape[1]):

        if y_head[0,i]<= 0.5:

            Y_prediction[0,i] = 0

        else:

            Y_prediction[0,i] = 1



    return Y_prediction

# predict(parameters["weight"],parameters["bias"],x_test)
def logistic_regression(X_train, y_train, X_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  X_train.shape[0]  # that is 4096

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, X_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],X_test)

    y_prediction_train = predict(parameters["weight"],parameters["bias"],X_train)



    # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(X_train, y_train, X_test, y_test,learning_rate = 0.01, num_iterations = 150)
#load dataset

#X is image array, y is label array

X = np.load('../input/Sign-language-digits-dataset/X.npy')

y = np.load('../input/Sign-language-digits-dataset/Y.npy')
#We use 0 and 1s; so create arrays for this:

#from 204 to 409 are zero sign, 822 to 1027 are one sign

X = np.concatenate((X[204:409], X[822:1027]), axis = 0)



print(X.shape)



zeros = np.zeros(205)

ones = np.ones(205)

y = np.concatenate((zeros, ones), axis=0).reshape(X.shape[0], 1)

print(y.shape)
#In order to use as input, we need to flatten the shape of X:

X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])

print(X.shape)



#410,4096 means that X data set have 410 samples and 4096 features
#Import function

from sklearn.model_selection import train_test_split



#Split data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
from sklearn import linear_model

logreg = linear_model.LogisticRegression(random_state = 42,max_iter= 150)

print("test accuracy: {} ".format(logreg.fit(X_train, y_train).score(X_test, y_test)))

print("train accuracy: {} ".format(logreg.fit(X_train, y_train).score(X_train, y_train)))