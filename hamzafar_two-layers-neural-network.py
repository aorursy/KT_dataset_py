# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # ploting graph
def generate_bits(n_x, m):
# Generate a m x n_x array of ints between 0 and 1, inclusive:
# m: number of rows
# n_x : number of columns per rows/ feature set
    np.random.seed(1)
    data = np.random.randint(2, size=(n_x, m))
    return(data)

def generate_label(data, m):
    # generate label by appyling xor operation to individual row
    # return list of label (results)
        # data: binary data set of m by n_x size
    lst_y = []
    y= np.empty((m,1))
    k = 0
    for tmp in data.T:
        xor = np.logical_xor(tmp[0], tmp[1])

        for i in range(2, tmp.shape[0]):
            xor = np.logical_xor(xor, tmp[i])
    #     print(xor)
        lst_y.append(int(xor))
        y[k,:] = int(xor)
        k+=1
    return(y.T)
def layer_sizes(x, y, hidden):
    # create neural network layers and return input, output and hidden units size
        # x : imput data
        # y : output labels
        # hidden : number of hidden units in hidden layer
    n_x = x.shape[0]
    n_h = hidden
    n_y = y.shape[0]
    
    return(n_x, n_h, n_y)
def initialize_param(n_x, n_h, n_y):
    # intialize w to random and b to zero and return paramters
        # n_x : number of input feature
        # n_h : number of neurons
        # n_y : number of output
    np.random.seed(15)
    
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.rand(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    
    parameters = {'w1' : w1,
             'w2' : w2,
             'b1' : b1,
             'b2' : b2
            }
    return(parameters)
def sigmoid(z):
    # Takes input as z and return sogmoid of value
    s = 1 / (1 + np.exp(-z))
    return s
def forward_propagte(x,y, parameters):
    # compute activations and return z and a
        # x: input data
        # y: target value
        # parameters: dictonary object of w's and b's
    z1 = np.dot(parameters['w1'], x) + parameters['b1']
    a1 = np.tanh(z1)
    z2 = np.dot(parameters['w2'], a1) + parameters['b2']
    a2 = sigmoid(z2)
    cache = {"z1": z1,
         "a1": a1,
         "z2": z2,
         "a2": a2}
    return(cache)
def compute_cost(y, parameters, cache, m):
    # calculate cost w.r.t. activation a2 and actual value y
        # y: target value
        # cache: dictionary of activations and z's
        # m: number of training examples in data set
    a = cache['a2']
    loss = -1*(y* np.log(a) + (1-y) * np.log(1-a))
    cost = np.sum(loss)/m
    return(cost)
def back_propagate(x, y, parameters, cache, m):
    # compute and return derivatives of paramters 
        # x: input data
        # y: target value
        # parameters: dictonary object of w's and b's
        # cache: dictionary of activations and z's
        # m: number of training examples in data set
        
    a2 =cache['a2']
    a1 =cache['a1']
    w2 = parameters['w2']

    dz2 = a2 - y

    dw2 = (1 / m) * np.dot(dz2, a1.T)
    db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

#     np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    
    dtan = 1 - np.power(a1, 2)
    dz1 = np.dot(w2.T, dz2) * dtan

    dw1 =(1 / m) * np.dot(dz1, x.T)
    db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)
    
    grads = {'dw1': dw1,
             'dw2': dw2,
             'db1': db1,
             'db2': db2,
        }
    return(grads)
def update_parameters(parameters, grads, lr):
    # update and return parameters 
        # parameters: dictonary object of w's and b's
        # grads: dictionary object of gradient of w's and b's
                
    parameters['w1'] = parameters['w1'] -(lr * grads['dw1'])
    parameters['w2'] = parameters['w2'] -(lr * grads['dw2'])

    parameters['b1'] = parameters['b1'] -(lr * grads['db1'])
    parameters['b2'] = parameters['b2'] -(lr * grads['db2'])

    return(parameters)
def optimize_parameters(x, y, parameters, m, num_iter):
    # This function will iterate according to desired number, and update the paramters
    # also return paramters and list of cost values.
        # x: input data
        # y: target value
        # parameters: dictonary object of w's and b's
        # m: number of training examples in data set
        #num_iter: number of iteration to update parameters and comoute cost
    lst_cost = []
    
    for i in range(num_iter):
        cache = forward_propagte(x,y, parameters)
    #     print('cost of ite:', compute_cost(y, parameters, cache, m))
        grads = back_propagate(x, y, parameters, cache, m)
        parameters = update_parameters(parameters, grads, lr)
        lst_cost.append(compute_cost(y, parameters, cache, m))
    return(parameters, lst_cost)
def plt_res(lst, ylab, lr):
    #This will plot the list of values at y axis while x axis will contain number of iteration
    #lst: lst of action/cost
    #ylab: y-axis label
    #lr: learning rate
    plt.plot(lst)
    plt.ylabel(ylab)
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(lr))
    plt.show()
n_x = 50
m = 10000
lr = 0.07
num_iter = 1000

x = generate_bits(n_x, m)
y = generate_label(x, m)

n_x, n_h, n_y = layer_sizes(x, y, 10)
parameters = initialize_param(n_x, n_h, n_y)

parameters, lst_cost_s = optimize_parameters(x, y, parameters, m, num_iter)

################################################################################
m = 100000

x = generate_bits(n_x, m)
y = generate_label(x, m)

n_x, n_h, n_y = layer_sizes(x, y, 10)
parameters = initialize_param(n_x, n_h, n_y)

parameters, lst_cost_m = optimize_parameters(x, y, parameters, m, num_iter)

print('------- 10000 training sample-------------')
plt_res(lst_cost_s, 'cost', lr)
print('------- 100000 training sample-------------')
plt_res(lst_cost_m, 'cost', lr)