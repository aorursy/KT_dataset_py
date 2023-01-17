# -*- coding: utf-8 -*-

"""

Created on Sun Sep  1 00:56:35 2019



@author: sazid

"""

import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn

import sklearn.datasets

import sklearn.linear_model

from sklearn.model_selection import train_test_split



def relu(Z):

    

    return np.maximum(0, Z)



def sigmoid(Z):

    

    Z = 1 / (1 + np.exp(-Z))

    #print(Z)

    #print("Z shape = "+ str(Z.shape))

    

    return Z



def propagate( w, b, X, Y):

    

    m = X.shape[1]

    # print(m)

    Z = np.dot( w.T, X) + b

    #print(Z.shape)

    #print(Z)

    

    A = relu(Z)

    #print('A shape = ' + str(A.shape))

    #print('Y shape' + str(Y.shape))

    #print(A)

    

    cost = (-1/m)*  np.sum( (Y * np.log(A)) + ((1 - Y)* np.log(1 - A))  ) 

    #print('first cost value = ' + str(cost))

    

    dZ = A - Y

    #print("dZ = " + str(dZ.shape))

    

    dw = (1/m)* np.dot(X, dZ.T)

    #print("dw  =" + str(dw.shape))

    

    db = (1/m)* np.sum(dZ)

    #print("db  =" + str(db.shape))

    #print("db = "+ str(db))

    

    assert(dw.shape == w.shape)

    assert(db.dtype == float)

    

    cost = np.squeeze(cost)

    assert(cost.shape == ())

    

    grads = {"dw": dw, "db" : db}

    

    

    return grads, cost



def optimize(w, b, X, Y, iteration, learning_rate, print_cost = True):

    

    costs = []

    

    for i in range(iteration):

        

        

        # Cost and gradient calculation 

        grads, cost = propagate(w , b ,X ,Y)

       

        # Retrieve derivatives from grads

        dw = grads["dw"]

        db = grads["db"]

        

        # update

        w = w - (learning_rate*dw)

        b = b - (learning_rate*db)

        

        # Record the costs

        if i % 100 == 0:

            costs.append(cost)

        

        # Print the cost every 100 training iterations

        if print_cost and i % 100 == 0:

            print ("Cost after iteration %i: %f" %(i, cost))

    

    params = {"w": w,

              "b": b}

    

    grads = {"dw": dw,

             "db": db}

    

    return params, grads, costs

    

def initialize_parameters(dim):

    

    #normal Init

    w = np.random.randn(dim, 1)*0.01

    #print(W.shape)

    #print(W)

    

    b = 0 

    #b = np.zeros((1,1))

    #print(b.shape)

    #print(b)

    

    # HE initialization

    #w = np.random.randn(dim , 1)*np.sqrt(2/960)

    #b = 0

    

    #Xavier initialization

    #w = np.random.randn(dim , 1)*np.sqrt(1/960)

    #b = 0

    

    assert(w.shape == (dim, 1))

    assert(isinstance(b, float) or isinstance(b, int))

    

    return w, b



def predict( w, b, X):

    

    m = X.shape[1]

    Y_prediction = np.zeros((1,m))

    w = w.reshape(X.shape[0], 1)

    

    Z = np.dot( w.T, X)

    #print("Z shape predict = " + str(Z.shape))

     

    A = relu(Z)

    #print("A shape" + str(A.shape))

    

    for i in range(A.shape[1]):

        

        # Convert probabilities A[0,i] to actual predictions p[0,i]

        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

        

        ### END CODE HERE ###

    

    assert(Y_prediction.shape == (1, m))

    

    return Y_prediction

    



def NN_model_zero(X_train, Y_train, X_test, Y_test, iteration, learning_rate):

    

    hidden_layer = 1;

    hidden_unit = 1;

    dim = 4

    w, b = initialize_parameters(dim)

    

    #print(w.shape)

    #print(b.shape)

    

    parameters, grads, costs = optimize(w, b, X_train, Y_train, iteration, learning_rate, print_cost = True)

    

    w = parameters["w"]

    b = parameters["b"]

    

    Y_prediction_test = predict(w , b, X_test)

    Y_prediction_train = predict(w, b, X_train)

    

    print("Y_prediction_train_shape = " + str(Y_prediction_train.shape))

    

     # Print train/test Errors

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))

    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

 

    d = {"costs": costs,

         "Y_prediction_test": Y_prediction_test, 

         "Y_prediction_train" : Y_prediction_train, 

         "w" : w, 

         "b" : b,

         "learning_rate" : learning_rate,

         "iteration": iteration}

    

    return d

    



def data_input_and_process_function():

    

     a = np.loadtxt("/kaggle/input/data_banknote_authentication.txt",delimiter=',')

     #print(a.shape)

     #print(a)

     

     #checking nan values

     #print(np.isnan(a).any())

     

     y = np.array(a[:,4]).reshape(1372,1)

     #print(y.shape)

     #print(y)

     

     a = np.delete(a, 4, 1)

     #print(a.shape)

     #print(a)

     

     train_set_X, test_set_X, train_set_Y, test_set_Y = train_test_split(a, y, test_size=0.3)

     #print(train_set_X.shape)

     #print(test_set_X.shape)

     #print(train_set_Y.shape)

     #print(test_set_Y.shape)

     

     train_set_X = train_set_X.reshape(4, 960)

     test_set_X = test_set_X.reshape(4, 412)

     train_set_Y = train_set_Y.reshape(1, 960)

     test_set_Y = test_set_Y.reshape(1, 412)

     

     #print(train_set_X.shape)

     #print(test_set_X.shape)

     #print(train_set_Y.shape)

     #print(test_set_Y.shape)

     

     #Normalize

     train_set_X = train_set_X - np.mean(train_set_X)

     train_set_X = train_set_X / np.std(train_set_X)

     

     test_set_X = test_set_X - np.mean(test_set_X)

     test_set_X = test_set_X / np.std(test_set_X)

        

     return train_set_X, test_set_X, train_set_Y, test_set_Y

    

def main_func_call():

    

    train_set_X, test_set_X, train_set_Y, test_set_Y = data_input_and_process_function()

    d = NN_model_zero(train_set_X, train_set_Y, test_set_X, test_set_Y, iteration = 9500, learning_rate = 0.001)

    

    costs = np.squeeze(d['costs'])

    

    plt.plot(costs)

    plt.ylabel('cost')

    plt.xlabel('iterations (per hundreds)')

    plt.title("Learning rate =" + str(d["learning_rate"]))

    plt.show()



main_func_call()    


