import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/data.csv")
data.info()
data.drop(['Unnamed: 32'], axis=1, inplace=True)
data.head()
data.drop(['id'], axis=1, inplace=True)
data.diagnosis.value_counts()
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x = data.drop(['diagnosis'], axis=1)
x = (x - np.min(x) ) / ( np.max(x) - np.min(x) ).values
f, axis = plt.subplots(figsize = (18,18))
sns.heatmap(data.corr(), annot = False, linewidths = .4, ax = axis)
plt.show()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w, b
def sigmoid(z):
    output = 1/( 1 + np.exp(-z) )
    return output
def forward_backward_propagation(w,b,x_train,y_train):
    y_ = np.dot(w.T,x_train) + b # numeric output of regression algorithm
    y_pre = sigmoid(y_) # binary output of sigmoid function
    loss = -y_train*np.log(y_pre)-(1-y_train)*np.log(1-y_pre) # output of loss function
    cost = (np.sum(loss))/x_train.shape[1] # x_train.shape[1]  is for scaling
def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    y_ = np.dot(w.T,x_train) + b # numeric output of regression algorithm
    y_pre = sigmoid(y_) # binary output of sigmoid function
    loss = -y_train*np.log(y_pre)-(1-y_train)*np.log(1-y_pre) # output of loss function
    cost = (np.sum(loss))/x_train.shape[1] # x_train.shape[1]  is for scaling
    # backward propagation
    derivative_weight = (np.dot(x_train,((y_pre-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_pre-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients = {"derivative_weight": derivative_weight,"derivative_bias": derivative_bias}
    return cost,gradients
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
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    
    dimension =  x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    
logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 100) 