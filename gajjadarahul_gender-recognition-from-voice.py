# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/voice.csv')
data.head(5)
data.label.unique
print(data.info())
data.label = [1 if each=="female" else 0 for each in data.label]
data.label.values
y = data.label.values
x_data = data.drop(['label'],axis=1)
np.min(x_data)
np.max(x_data)
x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.head(5)
x_data.head(5)
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

#Now lets transpose the above all
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

print("x_train shape : ", x_train.shape)
print("x_test shape : ", x_test.shape)
print("y_train shape : ", y_train.shape)
print("y_test shape : ", y_test.shape)
def initialization(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

def f_b_propagation(w,b,x_train, y_train):
    #forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -( y_train * np.log(y_head)- ( 1 - y_train ) * np.log( 1 - y_head))
    cost = (np.sum(loss))/x_train.shape[1]   # x_train.shape[1]is for scaling the cost
    
    #backward propagation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T)))/x_train.shape[1]
    #x_train.shape[1] is for scaling purpose
    
    derivative_bias = np.sum(y_head - y_train)/ x_train.shape[1]
    #x_train.shape[1] is for scaling purpose
    
    gradients = {"Derivative_Weight": derivative_weight, "Derivative_Bias": derivative_bias}
    
    return cost, gradients
def update(w, b, x_train, y_train, learning_rate, number_of_iterations):
    cost_list_1 = []
    cost_list_2 = []
    index = []
    
    #Iterate over the updating of parameters
    for i in range(number_of_iterations):
        #Make forward and backward propagation to find the cost and gradient
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list_1.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 10 == 0:
            cost_list_2.append(cost)
            index.append(i)
            print("Cost after iteration %i =  %f " %(i, cost))
        
