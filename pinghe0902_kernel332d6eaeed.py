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
# import librayies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import pandas as pd

#np.set_printoptions(threshold=np.inf)
# Read data from csv files

df_train = pd.read_csv('../input/train.csv')
train_all = df_train.values

df_test = pd.read_csv('../input/test.csv')
test_all = df_test.values
# transform csv data to matrix for visualization

train_x_orig = train_all[:, 1:(train_all.shape[1])].reshape(42000, 28, 28)
train_y_orig = train_all[:, 0].reshape(42000, 1)
print("dims of train_x: ", str(train_x_orig.shape))
print("dims of train_y: ", str(train_y_orig.shape))

test_x_orig = test_all.reshape(28000, 28, 28)
test_y_orig = np.zeros((28000, 1))
print("dims of test_x_orig: ", str(test_x_orig.shape))
print("dims of test_y_orig: ", str(test_y_orig.shape))
index = 40
plt.imshow(train_x_orig[index])
plt.figure(1)
print(train_y_orig[index])
plt.figure(2)
plt.imshow(test_x_orig[index])
print(test_y_orig[index])
# transform csv data to matirx for computation

train_set_x_all = train_all[:, 1:train_all.shape[1]].T
train_set_y_all = train_y_orig.T
test_set_x = test_all.T
test_set_y = test_y_orig.T

print("train_set_x_all: ", str(train_set_x_all.shape))
print("train_set_y_all: ", str(train_set_y_all.shape))
print("test_set_x: ", str(test_set_x.shape))
print("test_set_y: ", str(test_set_y.shape))
# Divides train set into train set and dev set

train_set_x_1 = train_set_x_all[:, 0:40000]
train_set_y_1 = train_set_y_all[:, 0:40000]

dev_set_x_1 = train_set_x_all[:, 40000:42000]
dev_set_y_1 = train_set_y_all[:, 40000:42000]

print("train_set_x_1: ", str(train_set_x_1.shape))
print("train_set_y_1: ", str(train_set_y_1.shape))
print("dev_set_x_1: ", str(dev_set_x_1.shape))
print("dev_set_y_1: ", str(dev_set_y_1.shape))
# Standardize data set x

train_set_x_all = train_set_x_all / 255
train_set_x = train_set_x_1 / 255
dev_set_x = dev_set_x_1 / 255
test_set_x = test_set_x / 255

print("train_set_x_all: ", str(train_set_x_all.shape))
print("train_set_x: ", str(train_set_x.shape))
print("dev_set_x: ", str(dev_set_x.shape))
print("test_set_x: ", str(test_set_x.shape))
# Neural Network Function: onehot

def onehot(y):
    
    length = y.shape[1]
    maxNum = np.squeeze(np.max(y, axis = 1))
    output = np.zeros((int(maxNum + 1), length))
    
    for i in range(length):
        
        output[int(y[0, i]), i] = 1
        
    return output
# Encoding data set y to onehot

train_set_y_all = onehot(train_set_y_all)
train_set_y = onehot(train_set_y_1)
dev_set_y = onehot(dev_set_y_1)
print("train_set_y.shape = " + str(train_set_y.shape))
print("dev_set_y.shape = " + str(dev_set_y.shape))

# Neural Network Function: mini_batch_data_set

def mini_batch_data_set(x, y, batch_size):
    
    error_flag = False
    batch_number = x.shape[1] // batch_size
    batch_last = x.shape[1] % batch_size
    X = []
    Y = []
    
    if batch_last == x.shape[1]:
        error_flag = True
        
    # shuffle train set data
    shuffled_permutation = np.random.permutation(x.shape[1])        
    x = x[:, shuffled_permutation]
    y = y[:, shuffled_permutation]
        
    for i in range(batch_number):
        
        X.append(x[:, batch_size * i : batch_size * (i + 1)].reshape(x.shape[0], batch_size))
        Y.append(y[:, batch_size * i : batch_size * (i + 1)].reshape(y.shape[0], batch_size))
    
    if batch_last != 0:
        
        X.append(x[:, batch_size * (batch_number) : x.shape[1]])
        Y.append(y[:, batch_size * (batch_number) : x.shape[1]])
    
    if batch_last != 0:
        return X, Y, batch_number + 1, error_flag
    else:
        return X, Y, batch_number, error_flag
# Neural Network Function: ReLu activation function

def relu(z):
    
    a = np.maximum(z, 0)
    
    return a

# Neural Network function: sigmoid activation fucntion

def sigmoid(z):
    
    a = 1 / (1 + np.exp(-z))
    
    return a
# Neural Network Function: tanh activation function

def tanh(z):
    
    a = (1 - np.exp(-z)) / (1 + np.exp(-z))
    
    return a
# Neural Network Function: Softmax activation function

def softmax(z):
    
    temp = np.exp(z)
    a = temp / np.sum(temp, axis = 0, keepdims = True)
    
    return a
# Neural Network Function: Initialize parameters

def initialize_parameters(layers):
    
    length = len(layers)
    parameters = {}
    
    for i in range(1, length):
        
        parameters['w' + str(i)] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2. / layers[i - 1])
        parameters['b' + str(i)] = np.zeros((layers[i], 1))
        
        assert(parameters['w' + str(i)].shape == (layers[i], layers[i-1]))
        assert(parameters['b' + str(i)].shape == (layers[i], 1))
    
    return parameters
    
# Neural Network Function: Forward propagation

def forward_propagation(train_set_x, parameters, layers, keep_prob, drop_out_isEnabled, activation, mini_batch_isEnabled, mini_batch_number):
    
    caches = {}
    drop_out = {}
    n = len(layers)
        
    if mini_batch_isEnabled == True:
        caches['A0'] = train_set_x[mini_batch_number]
    else:
        caches['A0'] = train_set_x
        
    if drop_out_isEnabled == True:
        drop_out['d0'] = np.random.rand(caches['A0'].shape[0], caches['A0'].shape[1]) < keep_prob[0]
        caches['A0'] = np.multiply(caches['A0'], drop_out['d0'])
        caches['A0'] = np.divide(caches['A0'], keep_prob[0])

    for i in range(1, n):

        caches['Z' + str(i)] = np.dot(parameters['w' + str(i)], caches['A' + str(i - 1)]) + parameters['b' + str(i)]

        if i < n - 1:
            if activation == 'sigmoid':
                caches['A' + str(i)] = sigmoid(caches['Z' + str(i)])
            elif activation == 'relu':
                caches['A' + str(i)] = relu(caches['Z' + str(i)])
            elif activation == 'tanh':
                caches['A' + str(i)] = tanh(caches['Z' + str(i)])
            else:
                print("activation function error !!!")

        else:
            if layers[n - 1] == 1:
                caches['A' + str(i)] = sigmoid(caches['Z' + str(i)]).reshape(1, -1) 
            else:
                caches['A' + str(i)] = softmax(caches['Z' + str(i)])
                
        if drop_out_isEnabled == True:
            drop_out['d' + str(i)] = np.random.rand(caches['A' + str(i)].shape[0], caches['A' + str(i)].shape[1]) < keep_prob[i]
            caches['A' + str(i)] = np.multiply(caches['A' + str(i)], drop_out['d' + str(i)])
            caches['A' + str(i)] = np.divide(caches['A' + str(i)], keep_prob[i])

        
    return caches
        
###### Neural Network Function: compute_cost

def compute_cost(parameters, caches, Y, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number):
    
    layers = len(caches) // 2
    A = caches['A' + str(layers)]
    
    if mini_batch_isEnabled == True:
        Y = Y[mini_batch_number]
    else:
        pass
    
    m = Y.shape[1]
    n_Y = Y.shape[0]
    
    if n_Y == 1:
        L = - (np.dot(Y, np.log(A).T) + np.dot((1 - Y), np.log(1 - A).T))
                
    else:
        L = - np.sum(np.multiply(Y, np.log(A)), axis = 0, keepdims = True)
        
    L2_regularization = 0;
    if L2_regularization_isEnabled == True:
        for i in range(1, layers + 1):
            L2_regularization = L2_regularization + (lambd / (2 * m)) * np.square(np.linalg.norm(parameters['w' + str(i)], ord = 'fro'))

    cost = (1 / m) * np.sum(L) + L2_regularization
    
    return cost
# Neural Network Function: compute derivative of activation function

def derivative(a, activation):
    '''
    a: input data, matrix
    activation: activation function
    '''
    
    if (activation == 'sigmoid'):
        results = np.multiply(a, 1 - a)
    elif (activation == 'relu'):
        results = a > 0
    elif (activation == 'tanh'):
        results = 1 - np.power(a, 2)
    else:
        None
    
    return results
# Neural Network Function: Backward propagation

def backward_propagation(train_set_x, train_set_y, parameters, caches, layers, activation, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number):
    
    grads = {}
    n = len(layers) - 1
    
    if mini_batch_isEnabled == True:
        train_set_y = train_set_y[mini_batch_number]  
    
    m = train_set_y.shape[1]
    grads['dZ' + str(n)] = caches['A' + str(n)] - train_set_y
    
    assert(grads['dZ' + str(n)].shape == (layers[n], m))
    
    for i in range(n, 0, -1):
        
        if L2_regularization_isEnabled == True:
            L2_regularization = (lambd / m) * parameters['w' + str(i)]
        else:
            L2_regularization = 0
        
        grads['dw' + str(i)] = (1 / m) * np.dot(grads['dZ' + str(i)], caches['A' + str(i - 1)].T) + L2_regularization
        grads['db' + str(i)] = (1 / m) * np.sum(grads['dZ' + str(i)], axis = 1, keepdims = True).reshape(layers[i], 1) 
        
        if (i > 1):
            grads['dA' + str(i - 1)] = np.dot(parameters['w' + str(i)].T, grads['dZ' + str(i)])
            grads['dZ' + str(i - 1)] = np.multiply(grads['dA' + str(i - 1)], derivative(caches['A' + str(i - 1)], activation))
        
            assert(grads['dA' + str(i - 1)].shape == (layers[i - 1], m))
            assert(grads['dZ' + str(i - 1)].shape == (layers[i - 1], m))
            
        assert(grads['dw' + str(i)].shape == (layers[i], layers[i - 1]))
        assert(grads['db' + str(i)].shape == (layers[i], 1))
        assert(grads['dw' + str(i)].shape == parameters['w' + str(i)].shape)
        assert(grads['db' + str(i)].shape == parameters['b' + str(i)].shape)
        
    return grads
    
# Neural Network Function: optimize_algorithm

def optimize_algorithm(parameters, layers, learning_rate, grads, algorithm, t):
    
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-8
    
    n = len(layers)
    
    if algorithm == 'gradient_decent':
        
        for i in range(1, n):
            
            parameters['w' + str(i)] = parameters['w' + str(i)] - learning_rate * grads['dw' + str(i)]
            parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * grads['db' + str(i)]
        
    elif algorithm == 'Adam':
        
        V_corrected = {}
        S_corrected = {}
        
        for i in range(1, n):
            
            parameters['Vdw' + str(i)] = beta_1 * parameters['Vdw' + str(i)] + (1 - beta_1) * grads['dw' + str(i)]
            parameters['Vdb' + str(i)] = beta_1 * parameters['Vdb' + str(i)] + (1 - beta_1) * grads['db' + str(i)]
            
            parameters['Sdw' + str(i)] = beta_2 * parameters['Sdw' + str(i)] + (1 - beta_2) * np.square(grads['dw' + str(i)])
            parameters['Sdb' + str(i)] = beta_2 * parameters['Sdb' + str(i)] + (1 - beta_2) * np.square(grads['db' + str(i)])
            
            V_corrected['Vdw' + str(i)] = parameters['Vdw' + str(i)] / (1 - np.power(beta_1, t))
            V_corrected['Vdb' + str(i)] = parameters['Vdb' + str(i)] / (1 - np.power(beta_1, t))
            
            S_corrected['Sdw' + str(i)] = parameters['Sdw' + str(i)] / (1 - np.power(beta_2, t))
            S_corrected['Sdb' + str(i)] = parameters['Sdb' + str(i)] / (1 - np.power(beta_2, t))
            
            parameters['w' + str(i)] = parameters['w' + str(i)] - learning_rate * V_corrected['Vdw' + str(i)] / (np.sqrt(S_corrected['Sdw' + str(i)]) + epsilon)
            parameters['b' + str(i)] = parameters['b' + str(i)] - learning_rate * V_corrected['Vdb' + str(i)] / (np.sqrt(S_corrected['Sdb' + str(i)]) + epsilon)
        
    return parameters
# Neural Network Function: gradient check

def gradient_check(train_set_x, 
                   train_set_y, 
                   parameters, 
                   layers, 
                   lambd, 
                   keep_prob, 
                   drop_out_isEnabled = False, 
                   L2_regularization_isEnabled = False, 
                   activation = 'relu', 
                   epsilon = 1e-7, 
                   mini_batch_isEnabled = False, 
                   mini_batch_size = 1, 
                   mini_batch_number = 1):
    
    caches_epsilonPlus = {}
    caches_epsilonMinus = {}
    grads_Approximation = []
    grads_BackPropagation = []
    
    n = len(layers)
    times = 0
    
    for i in range(1, n):
        
        #print("i = " + str(i))
        row = parameters['w' + str(i)].shape[0]
        column = parameters['w' + str(i)].shape[1]
        
        for r in range(row):
            for c in range(column):
                
                times = times + 1
                #print("At times " + str(times))
                #print("w(" + str(i) + ")[" + str(r) + ", " + str(c) + "]: ")
                
                parameters_epsilonPlus = copy.deepcopy(parameters)
                parameters_epsilonMinus = copy.deepcopy(parameters)

                
                #print("parameters_epsilonPlus = " + str(parameters_epsilonPlus))
                #print("parameters_epsilonMinus = " + str(parameters_epsilonMinus))
                
                #print("parameters_epsilonPlus['w" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonPlus['w' + str(i)][r, c]))
                #print("parameters_epsilonMinus['w" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonMinus['w' + str(i)][r, c]))
                #print("epsilon = " + str(epsilon))
                
        
                parameters_epsilonPlus['w' + str(i)][r, c] += epsilon
                parameters_epsilonMinus['w' + str(i)][r, c] -= epsilon
                
                
                #print("After plus, parameters_epsilonPlus['w" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonPlus['w' + str(i)][r, c]))
                #print("After minus, parameters_epsilonMinus['w" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonMinus['w' + str(i)][r, c]))
                #print("parameters['w" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters['w' + str(i)][r, c]))
                

                caches_epsilonPlus = forward_propagation(train_set_x, parameters_epsilonPlus, layers, keep_prob, drop_out_isEnabled, activation, mini_batch_isEnabled, mini_batch_number)
                caches_epsilonMinus = forward_propagation(train_set_x, parameters_epsilonMinus, layers, keep_prob, drop_out_isEnabled, activation, mini_batch_isEnabled, mini_batch_number)

                
                #print("caches_epsilonPlus = " + str(caches_epsilonPlus))
                #print("caches_epsilonMinus = " + str(caches_epsilonMinus))
                
                #print(caches_epsilonPlus['Z2'][0, 0])
                
                
                cost_epsilonPlus = compute_cost(parameters, caches_epsilonPlus, train_set_y, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number)
                cost_epsilonMinus = compute_cost(parameters, caches_epsilonMinus, train_set_y, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number)    
                
                
                #print("cost_epsilonPlus = " + str(cost_epsilonPlus))
                #print("cost_epsilonMinus = " + str(cost_epsilonMinus))
                

                grad_Approximation = (cost_epsilonPlus - cost_epsilonMinus) / (2 * epsilon)
                grads_Approximation.extend(grad_Approximation.reshape(-1))
                #print("grad_Approximation is " + str(grad_Approximation))
                
        #print("At times " + str(times) + ", length of grads_Approximation = " + str(len(grads_Approximation)))
        row = parameters['b' + str(i)].shape[0]
        column = parameters['b' + str(i)].shape[1]
        
        for r in range(row):
            for c in range(column):   
                
                times = times + 1
                #print("At times " + str(times))
                #print("b(" + str(i) + ")[" + str(r) + ", " + str(c) + "]: ")
                
                parameters_epsilonPlus = copy.deepcopy(parameters)
                parameters_epsilonMinus = copy.deepcopy(parameters)
                
                
                #print("parameters_epsilonPlus['b" + str(i) + "'] = " + str(parameters_epsilonPlus))
                #print("parameters = " + str(parameters))
                
                
                #print("parameters_epsilonPlus['b" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonPlus['b' + str(i)][r, c]))
                #print("parameters_epsilonMinus['b" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonMinus['b' + str(i)][r, c]))
                #print("epsilon = " + str(epsilon))
                

                parameters_epsilonPlus['b' + str(i)][r, c] += epsilon
                parameters_epsilonMinus['b' + str(i)][r, c] -= epsilon
                
                
                #print("parameters_epsilonPlus['b" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonPlus['b' + str(i)][r, c]))
                #print("parameters_epsilonMinus['b" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters_epsilonMinus['b' + str(i)][r, c]))
                #print("parameters['b" + str(i) + "'][" + str(r) + ", " + str(c) + "] = " + str(parameters['b' + str(i)][r, c]))
                

                caches_epsilonPlus = forward_propagation(train_set_x, parameters_epsilonPlus, layers, keep_prob, drop_out_isEnabled, activation, mini_batch_isEnabled, mini_batch_number)
                caches_epsilonMinus = forward_propagation(train_set_x, parameters_epsilonMinus, layers, keep_prob, drop_out_isEnabled, activation, mini_batch_isEnabled, mini_batch_number)
                
                
                #print("caches_epsilonPlus = " + str(caches_epsilonPlus))
                #print("caches_epsilonMinus = " + str(caches_epsilonMinus))
                

                cost_epsilonPlus = compute_cost(parameters, caches_epsilonPlus, train_set_y, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number)
                cost_epsilonMinus = compute_cost(parameters, caches_epsilonMinus, train_set_y, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number)    
                
                
                #print("cost_epsilonPlus = " + str(cost_epsilonPlus))
                #print("cost_epsilonMinus = " + str(cost_epsilonMinus))
                

                grad_Approximation = (cost_epsilonPlus - cost_epsilonMinus) / (2 * epsilon)
                grads_Approximation.extend(grad_Approximation.reshape(-1))
                #print("grad_Approximation is " + str(grad_Approximation))
        
    grads_Approximation = grads_Approximation
    #print("grads_Approximation = " + str(grads_Approximation))
    
    caches = forward_propagation(train_set_x, parameters, layers, keep_prob, drop_out_isEnabled, activation, mini_batch_isEnabled, mini_batch_number)
    grads = backward_propagation(train_set_x, train_set_y, parameters, caches, layers, activation, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number)

    #print("caches = " + str(caches))
    #print("grads = " + str(grads))
    
    for i in range(1, n):
        
        #print("grads['dw" + str(i) + "'] = " + str(grads['dw' + str(i)]))
        grads_BackPropagation.extend(grads['dw' + str(i)].reshape(-1))
        #print("grads_BackPropagation = " + str(grads_BackPropagation))
        
        #print("grads['db" + str(i) + "'] = " + str(grads['db' + str(i)]))
        grads_BackPropagation.extend(grads['db' + str(i)].reshape(-1))
        #print("grads_BackPropagation = " + str(grads_BackPropagation))
    
    #print("grads_BackPropagation = " + str(grads_BackPropagation))
    
    numerator = np.linalg.norm(np.array(grads_Approximation) - np.array(grads_BackPropagation), ord = 2)
    #print("numerator of difference is " + str(numerator))
    
    denominator1 = np.linalg.norm(np.array(grads_Approximation), ord = 2)
    denominator2 = np.linalg.norm(np.array(grads_BackPropagation), ord = 2)
    #print("denominator1 of difference is " + str(denominator1))
    #print("denominator2 of difference is " + str(denominator2))
    
    difference = numerator / (denominator1 + denominator2)
    print("difference = " + str(difference))

    if difference < epsilon * 10:
        isPassed = True
    else:
        isPassed = False
    
    return isPassed
# Neural Network Function: predict

def predict(test_set_x, 
            parameters, 
            layers, 
            keep_prob, 
            drop_out_isEnabled = False, 
            activation = 'relu', 
            mini_batch_isEnabled = False, 
            mini_batch_number = 1):
    
    length = len(layers)
    
    caches = forward_propagation(test_set_x, 
                                 parameters, 
                                 layers, 
                                 keep_prob, 
                                 drop_out_isEnabled, 
                                 activation, 
                                 mini_batch_isEnabled = False, 
                                 mini_batch_number = 1)
    
    AL = caches['A' + str(length - 1)]
    
    if layers[length - 1] == 1:
        predictions = AL > 0.5
    else:
        max_array = np.where(AL == np.amax(AL, axis = 0))
        temp = np.array([max_array[0], max_array[1]]).T
        temp = temp[np.lexsort(temp.T)]
        predictions = temp[:, 0]
        
    return predictions
    
# Neural Network Function: model

def model(train_set_x, 
          train_set_y, 
          layers, 
          iterations, 
          learning_rate, 
          parameters_initialization,
          lambd, 
          L2_regularization_isEnabled, 
          keep_prob, 
          drop_out_isEnabled = False, 
          activation = 'relu', 
          algorithm = 'gradient_decent', 
          mini_batch_isEnabled = False, 
          mini_batch_size = 1, 
          gradient_check_isEnabled = False, 
          gradient_check_iter = 0, 
          epsilon = 1e-7):
    
    t = 0
    costs = []
    caches = {}
    grads = {}
    
    if parameters_initialization == 0:
        parameters = initialize_parameters(layers)
    else:
        parameters = parameters_initialization
    
    if algorithm == 'Adam':

        for j in range(1, len(layers)):
            parameters['Vdw' + str(j)] = np.zeros(parameters['w' + str(j)].shape)
            parameters['Vdb' + str(j)] = np.zeros(parameters['b' + str(j)].shape)
            parameters['Sdw' + str(j)] = np.zeros(parameters['w' + str(j)].shape)
            parameters['Sdb' + str(j)] = np.zeros(parameters['b' + str(j)].shape)
    
    #rint(parameters)
    for i in range(iterations):
                
            #-----------------------------------Gradient Check-----------------------------------

            if gradient_check_isEnabled == True and i == gradient_check_iter:

                if mini_batch_isEnabled == True:
                    train_set_x_check = train_set_x[0]
                    train_set_y_check = train_set_y[0]
                else:
                    train_set_x_check = train_set_x
                    train_set_y_check = train_set_y

                isPassed = gradient_check(train_set_x_check, 
                                          train_set_y_check, 
                                          parameters, 
                                          layers, 
                                          lambd = 0, 
                                          keep_prob = np.ones(layers.shape), 
                                          drop_out_isEnabled = False, 
                                          L2_regularization_isEnabled = False, 
                                          activation = 'relu', 
                                          epsilon = 1e-7)

                if isPassed == True:
                    print("Congratulation! Gradient checking passed!")
                else:
                    print("Error! Gradient checking not passed!")
                    return -1
                
            #-----------------------------------------End--------------------------------------------
            
            if mini_batch_isEnabled == True:
                
                mini_batch_train_set_x, mini_batch_train_set_y, mini_batch_numbers, error_flag = mini_batch_data_set(train_set_x, train_set_y, mini_batch_size)
                if error_flag == True:
                    print("Error! batch_size is overflowed!!")
                    return -1
            else:
                mini_batch_train_set_x = train_set_x
                mini_batch_train_set_y = train_set_y
                mini_batch_numbers = 1
                
            #print("mini_batch_train_set_x = " + str(mini_batch_train_set_x))
            #print("mini_batch_train_set_y = " + str(mini_batch_train_set_y))
            
            for mini_batch_number in range(mini_batch_numbers):

                t += 1
                #rint(mini_batch_number)
                caches = forward_propagation(mini_batch_train_set_x, parameters, layers, keep_prob, drop_out_isEnabled, activation, mini_batch_isEnabled, mini_batch_number)
                #rint(caches)
                cost = compute_cost(parameters, caches, mini_batch_train_set_y, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number)
                #rint(cost)
                grads = backward_propagation(mini_batch_train_set_x, mini_batch_train_set_y, parameters, caches, layers, activation, lambd, L2_regularization_isEnabled, mini_batch_isEnabled, mini_batch_number)
                #rint(grads)
                parameters = optimize_algorithm(parameters, layers, learning_rate, grads, algorithm, t)
                #rint(parameters)
                
            if i % 10 == 0:
                print("cost in iteration {} is: {}".format(i, cost))

            costs.append(cost)
        
    plt.figure(1)
    plt.plot(costs)
    plt.show()
    
    return parameters
        
#'''
layers = np.array([train_set_x.shape[0], 100, 20, 10])

parameters = model(train_set_x, 
                   train_set_y, 
                   layers, 
                   iterations = 100, 
                   learning_rate = 0.1, 
                   parameters_initialization = 0,
                   lambd = 0.1,  
                   L2_regularization_isEnabled = True, 
                   keep_prob = np.ones(layers.shape), 
                   drop_out_isEnabled = False, 
                   activation = 'relu', 
                   algorithm = 'gradient_decent', 
                   mini_batch_isEnabled = True, 
                   mini_batch_size = 50, 
                   gradient_check_isEnabled = False, 
                   gradient_check_iter = 0, 
                   epsilon = 1e-7)

predictions_train = predict(train_set_x, 
                            parameters, 
                            layers, 
                            keep_prob = np.array([1, 1, 1, 1]), 
                            drop_out_isEnabled = False, 
                            activation = 'relu')
accuracy = np.mean(predictions_train == train_set_y_1[0, :])
print("train_set_x's accuracy is " + str(accuracy))


predictions_dev = predict(dev_set_x, 
                          parameters, 
                          layers, 
                          keep_prob = np.array([1, 1, 1, 1]), 
                          drop_out_isEnabled = False, 
                          activation = 'relu')
accuracy = np.mean(predictions_dev == dev_set_y_1[0, :])
print("dev_set_x's accuracy is " + str(accuracy))
#'''
predictions_test = predict(test_set_x, parameters, layers, keep_prob = np.array([1, 1, 1, 1]), drop_out_isEnabled = False, activation = 'relu')

index = 400
plt.imshow(test_x_orig[index])
predictions_test[index]
submit = pd.DataFrame({"ImageId": [i for i in range(1, 28001)], "Label": predictions_test})
submit.to_csv("submission.csv", index = False)