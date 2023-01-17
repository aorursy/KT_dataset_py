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
# Package imports
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
    return img.resize((64,64), Image.ANTIALIAS)

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
train_output = np.concatenate((train_hot_dog_result,train_not_hot_dog_result),axis=1)
test_hot_dog = get_array('../input/seefood/test/hot_dog/')
test_not_hot_dog = get_array('../input/seefood/test/not_hot_dog/')
test_hot_dog_result = np.ones((1,test_hot_dog.shape[1]))
test_not_hot_dog_result = np.zeros((1,test_not_hot_dog.shape[1]))
test_input = np.concatenate((test_hot_dog,test_not_hot_dog),axis=1)
test_output = np.concatenate((test_hot_dog_result,test_not_hot_dog_result),axis=1)
train_input = train_input / 225.
test_input = test_input / 225.
print(train_input.shape)
print(train_output.shape)
print(test_input.shape)
print(test_output.shape)
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    A = np.maximum(0,Z)

    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ
def initialize_parameters_deep(layer_dims):
    
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
    
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        
    return parameters
def linear_forward(A, W, b):
    Z = np.dot(W,A)+b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    
    cache = (A, W, b)
    
    return Z, cache
def linear_activation_forward(A_prev, W, b, activation):
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    
    return A, cache
def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], "relu")
        caches.append(cache)
        
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches
def compute_cost(AL, Y):
    
    m = Y.shape[1]
    
    logprob = -1*np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    cost = (1/m)*logprob
    
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    return cost
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m)*np.dot(dZ,A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T,dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db
def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db
def L_model_backward(AL, Y, caches):
    
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, \
                                                                        current_cache, "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l+1)], \
                                                                        current_cache, "relu")
        
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    temp = parameters
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)]-learning_rate*grads["dW" + str(l + 1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)]-learning_rate*grads["db" + str(l + 1)]
    
    return parameters
layers_dims = [12288, 20, 7, 5, 1]
def L_layer_model(X, Y, layers_dims, learning_rate = 0.075, num_iterations = 3000, print_cost=False):
    costs = []
    
    parameters = initialize_parameters_deep(layers_dims)
    
    for i in range(0, num_iterations):
        
        AL, caches = L_model_forward(X, parameters)
        
        cost = compute_cost(AL, Y)
        
        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)
        
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost:
            costs.append(cost)
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
parameters = L_layer_model(train_input, train_output, layers_dims, learning_rate = 0.1, num_iterations = 2500, print_cost = True)
def predict(parameters, X, Y):
    AL, cache = L_model_forward(X, parameters)
    predictions = np.round(AL)
    print ('Accuracy: %d'% float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
predict(parameters, train_input, train_output)
predict(parameters, test_input, test_output)
