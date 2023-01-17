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
import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from os import listdir
from os.path import isfile, join

%matplotlib inline
def get_images(file_name):
    img = Image.open(file_name)
    return img.resize((256,256), Image.ANTIALIAS)

def get_file_list(folder):
    return [f for f in listdir(folder) if isfile(join(folder, f))]

def get_array(folder):
    image_list = get_file_list(folder)
    m = np.array([])
    for i_name in image_list:
        p = np.array(get_images(folder+i_name))
        p = p.reshape(p.shape[0]*p.shape[1]*p.shape[2],1)
        if len(m)==0:
            m = p
        else:
            m = np.concatenate((m,p),axis=1)
    return m
train_hot_dog = get_array('../input/seefood/train/hot_dog/')
train_not_hot_dog = get_array('../input/seefood/train/not_hot_dog/')
train_hot_dog_result = np.ones((1,train_hot_dog.shape[1]))
train_not_hot_dog_result = np.zeros((1,train_not_hot_dog.shape[1]))
train_input = np.concatenate((train_hot_dog,train_not_hot_dog),axis=1)
print(train_input.shape)
train_output = np.concatenate((train_hot_dog_result,train_not_hot_dog_result),axis=1)
print(train_output.shape)
test_hot_dog = get_array('../input/seefood/test/hot_dog/')
test_not_hot_dog = get_array('../input/seefood/test/not_hot_dog/')
test_hot_dog_result = np.ones((1,test_hot_dog.shape[1]))
test_not_hot_dog_result = np.zeros((1,test_not_hot_dog.shape[1]))
test_input = np.concatenate((test_hot_dog,test_not_hot_dog),axis=1)
test_output = np.concatenate((test_hot_dog_result,test_not_hot_dog_result),axis=1)
train_input = train_input / 225.
test_input = test_input / 225.
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w,b
def propagate(w,b,X,Y):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))
    
    # back
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    
    cost = np.squeeze(cost)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
    
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    costs = []
    
    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def predict(w, b, X):
    
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_with_zeros(X_train.shape[0])
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d
d = model(train_input, train_output, test_input, test_output, num_iterations = 2000, learning_rate = 0.0005, print_cost = True)