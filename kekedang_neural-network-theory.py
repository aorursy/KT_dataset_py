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
# Numpy array cannot be added as an argument.

def step_function(x):

    if x > 0:

        return 1

    else:

        return 0
# Numpy array can be added as an argument.

def step_function(x):

    y = x > 0 

    return y.astype(np.int)
# Specifically

import numpy as np

x = np.array([-2.0, 1.0, 2.0])

print(x)



y = x > 0

print(y) # bool array



y = y.astype(np.int) # bool to int

print(y)
# Step Function graph

import numpy as np

import matplotlib.pyplot as plt



def step_function(x):

    return np.array(x > 0, dtype=np.int)



x = np.arange(-6.0, 6.0, 0.1)

y = step_function(x)

plt.plot(x, y)

plt.ylim(-0.1, 1.1)

plt.show()
def sigmoid(x):

    return 1 / (1 + np.exp(-x))



# Handle numpy array properly 

x = np.array([-2.0, 1.0, 3.0])

print(sigmoid(x))
# Sigmoid function graph

x = np.arange(-6.0, 6.0, 0.1)

y = sigmoid(x)

plt.plot(x, y)

plt.ylim(-0.1, 1.1)

plt.show()
def relu(x):

    return np.maximum(0, x)
# Relu function graph

x = np.arange(-6.0, 6.0, 0.5)

y = relu(x)

plt.plot(x, y)

plt.ylim(-1, 6)

plt.show()
A = np.array([[1,2], [3,4]])

print(A.shape)



B = np.array([[5,6], [7,8]])

print(B.shape)



print(np.dot(A, B))
A = np.array([[2,3], [3,4], [5,6]])

print(A.shape)



B = np.array([8,9])

print(B.shape)



print(np.dot(A, B))
X = np.array([2,3])

print(X.shape)



W = np.array([[1,3,5], [2,4,6]])

print(W)

print(W.shape)



Y = np.dot(X, W)

print(Y)
X = np.array([2.0, 2.5])

W1 = np.array([[0.2, 0.4, 0.6], [0.3, 0.6, 0.8]])

B1 = np.array([-.2, 0.3, 0.4])



print(W1.shape)

print(X.shape)

print(B1.shape)



A1 = np.dot(X, W1) + B1

print(A1)
Z1 = sigmoid(A1)



print(A1)

print(Z1)
W2 = np.array([[0.2, 0.4], [0.3, 0.5], [0.4, 0.7]])

B2 = np.array([0.2, 0.3])



print(Z1.shape)

print(W2.shape)

print(B2.shape)



A2 = np.dot(Z1, W2) + B2

Z2 = sigmoid(A2)
def identity_function(x):

    return x



W3 = np.array([[0.2, 0.4], [0.3, 0.6]])

B3 = np.array([0.3,0.6])



A3 = np.dot(Z2, W3) + B3

Y = identity_function(A3) # Y = A3
# Organization

def init_network():

    network = {}

    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])

    network['b1'] = np.array([0.1, 0.2, 0.3])

    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])

    network['b2'] = np.array([0.1, 0.2])

    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])

    network['b3'] = np.array([0.1, 0.2])

    

    return network



def forward(network, x):

    W1, W2, W3 = network['W1'], network['W2'], network['W3']

    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    

    a1 = np.dot(x, W1) + b1

    z1 = sigmoid(a1)

    a2 = np.dot(z1, W2) + b2

    z2 = sigmoid(a2)

    a3 = np.dot(z2, W3) + b3

    y = identity_function(a3)

    

    return y



network = init_network()

x = np.array([1.0, 0.5])

y = forward(network, x)

print(y)

    
a = np.array([0.4, 3.0, 4.5])

exp_a = np.exp(a)

print(exp_a)



sum_exp_a = np.sum(exp_a)

print(sum_exp_a)



y = exp_a / sum_exp_a

print(y)
def softmax(a):

    exp_a = np.exp(a)

    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a



    return y
a = np.array([2000, 1900, 800])

print(np.exp(a) / np.sum(np.exp(a)))



c = np.max(a) # Maximum value of input signal

print(a - c)



print(np.exp(a - c) / np.sum(np.exp(a - c)))
def softmax(a):

    c = np.max(a)

    exp_a = np.exp(a - c) # Overflow Countermeasures

    sum_exp_a = np.sum(exp_a)

    y = exp_a / sum_exp_a



    return y
# You can interpret the output of a Softmax function as a probability

a = np.array([0.5, 3.0, 4.5])

y = softmax(a)

print(y)

print(np.sum(y))