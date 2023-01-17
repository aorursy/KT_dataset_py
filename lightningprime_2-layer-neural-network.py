# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#training set
data_train = pd.read_csv('../input/exoTrain.csv')
#test set
data_test = pd.read_csv('../input/exoTest.csv')
Y_train = np.reshape(np.array(data_train['LABEL']),(1,data_train.shape[0]))
X_train = np.transpose(np.array(data_train[data_train.columns[1:]]))
Y_test = np.reshape(np.array(data_test['LABEL']),(1,data_test.shape[0]))
X_test = np.transpose(np.array(data_test[data_test.columns[1:]]))
#Shapes of all X and Y for train and test set:
print("Shapes:")
print("Y_train = ",Y_train.shape)
print("X_train = ",X_train.shape)
print("Y_test = ",Y_test.shape)
print("X_test = ",X_test.shape)
# Normalization of both train and test set X:

#training set:
mean_train = np.reshape(np.mean(X_train,axis=0),(1,X_train.shape[1]))
std_train = np.reshape(np.std(X_train,axis=0),(1,X_train.shape[1]))
X_train = (X_train - mean_train)/std_train

#test set:
mean_test = np.reshape(np.mean(X_test,axis=0),(1,X_test.shape[1]))
std_test = np.reshape(np.std(X_test,axis=0),(1,X_test.shape[1]))
X_test = (X_test - mean_test)/std_test
#Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

#Derivative of Sigmoid Function
def derivatives_sigmoid(z):
    return z * (1 - z)
#Neural network structure
#Hidden layer's activation function = tanh()
#Output layer's activation function = sigmoid()
def defining_structure(X):
    n_i = X.shape[0]
    n_h = 1000
    n_o = 1
    
    nodes = {}
    nodes["n_i"]=n_i
    nodes["n_h"]=n_h
    nodes["n_o"]=n_o
    
    return nodes
#Total nodes in Hidden Layer
nodes = defining_structure(X_train)
print(nodes["n_h"])
def random_initialization(X):
    np.random.seed(2)
    
    nodes = defining_structure(X)
    
    hidden_weight = np.random.randn(nodes["n_h"],nodes["n_i"])
    hidden_bias = np.zeros((nodes["n_h"],1))
    output_weight = np.random.randn(nodes["n_o"],nodes["n_h"])
    output_bias = np.zeros((nodes["n_o"],1))
    
    parameters = {}
    parameters["hidden_weight"]=hidden_weight
    parameters["hidden_bias"]=hidden_bias
    parameters["output_weight"]=output_weight
    parameters["output_bias"]=output_bias
    
    return parameters
random_initialization(X_train)
def forward_propogation(X,Y,parameters):
    
    hidden_weight = parameters["hidden_weight"]
    hidden_bias = parameters["hidden_bias"]
    output_weight = parameters["output_weight"]
    output_bias = parameters["output_bias"]
    
    hidden_output = np.dot(hidden_weight,X) + hidden_bias
    hidden_activation = np.tanh(hidden_output)
    output_output = np.dot(output_weight,hidden_activation) + output_bias
    output_activation = np.tanh(output_output)
    output_activation = sigmoid(output_output)
    output_activation = np.abs(output_activation-0.0001)
    
    cache = {}
    cache["hidden_output"]=hidden_output
    cache["hidden_activation"]=hidden_activation
    cache["output_output"]=output_output
    cache["output_activation"]=output_activation
    
    return output_activation,cache
forward_propogation(X_train,Y_train,random_initialization(X_train))
def compute_cost(output_activation,Y):
    n = Y.shape[1]
    cost = -(np.dot(Y,np.transpose(np.log(output_activation))) + np.dot((1-Y),np.transpose(np.log(1-output_activation))))/n
    cost = np.squeeze(cost)
    return cost
def backward_propogation(X,Y,parameters,cache):
    n = Y.shape[1]
    hidden_weight = parameters["hidden_weight"]
    hidden_bias = parameters["hidden_bias"]
    output_weight = parameters["output_weight"]
    output_bias = parameters["output_bias"]
    
    hidden_output=cache["hidden_output"]
    hidden_activation=cache["hidden_activation"]
    output_output=cache["output_output"]
    output_activation=cache["output_activation"]

    d_output_activation = (-Y/output_activation) + (1-Y)/(1-output_activation)
    d_output_output = output_activation - Y
    d_output_weight = np.dot(d_output_output,np.transpose(hidden_activation))/n
    d_output_bias = np.sum(d_output_output,axis=1,keepdims=True)/n
    
    d_hidden_activation = np.dot(np.transpose(output_weight),d_output_output)      
    d_hidden_output = d_hidden_activation*(1-np.power(np.tanh(hidden_output),2))
    d_hidden_weight = np.dot(d_hidden_output,np.transpose(X))/n
    d_hidden_bias = np.sum(d_hidden_output,keepdims=True)/n
            
    gradients = {}
    gradients["d_hidden_weight"]=d_hidden_weight
    gradients["d_hidden_bias"]=d_hidden_bias
    gradients["d_output_weight"]=d_output_weight
    gradients["d_output_bias"]=d_output_bias
    
    return gradients
def update_weight(learning_rate,parameters,gradients):
    
    hidden_weight = parameters["hidden_weight"]
    hidden_bias = parameters["hidden_bias"]
    output_weight = parameters["output_weight"]
    output_bias = parameters["output_bias"]
    
    d_hidden_weight = gradients["d_hidden_weight"]
    d_hidden_bias = gradients["d_hidden_bias"]
    d_output_weight = gradients["d_output_weight"]
    d_output_bias = gradients["d_output_bias"]
    
    hidden_weight = hidden_weight - learning_rate*d_hidden_weight
    hidden_bias = hidden_bias - learning_rate*d_hidden_bias
    output_weight = output_weight - learning_rate*d_output_weight
    output_bias = output_bias - learning_rate*d_output_bias
    
    params = {}
    params["hidden_weight"]=hidden_weight
    params["hidden_bias"]=hidden_bias
    params["output_weight"]=output_weight
    params["output_bias"]=output_bias
    
    return params
def model(X,Y,learning_rate=0.1,epoch=3000):
    
    parameters = random_initialization(X)
    all_cost = list()
    all_accuracy = list()
        
    for i in range(epoch):
        output_activation,cache = forward_propogation(X,Y,parameters)
        cost = compute_cost(output_activation,Y)
        all_cost.append(cost)
        print("Cost for iteration ",i+1," = ",cost,end='\r')
        
        gradients = backward_propogation(X,Y,parameters,cache)
        parameters = update_weight(learning_rate,parameters,gradients)
        accuracy = np.squeeze(output_activation)
        accuracy = (100 - np.mean(np.abs(accuracy - Y))*100)
        all_accuracy.append(accuracy)
        
    nn_model = {}
    nn_model["gradients"]=gradients
    nn_model["cache"]=cache
    nn_model["parameters"]=parameters
    nn_model["cost"]=all_cost
    nn_model["accuracy"] = all_accuracy


    return nn_model
nn_model_train = model(X_train,Y_train)
nn_model_test = model(X_test,Y_test)
train_prediction = np.squeeze(nn_model_train["cache"]["output_activation"])
test_prediction = np.squeeze(nn_model_test["cache"]["output_activation"])
print("Training set accuracy = ",(100 - np.mean(np.abs(train_prediction - Y_train))*100))
print("Test set accuracy = ",(100 - np.mean(np.abs(test_prediction - Y_test))*100))
cost_train = np.squeeze(nn_model_train["cost"])
cost_test = np.squeeze(nn_model_test["cost"])

train, = plt.plot(cost_train,label='Training Loss',color='red')
test, = plt.plot(cost_test,label='Testing Loss',color='green')

plt.legend(handles=[train,test])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
accuracy_train = np.squeeze(nn_model_train["accuracy"])
accuracy_test = np.squeeze(nn_model_test["accuracy"])

train, = plt.plot(accuracy_train,label='Training Accuracy',color='red')
test, = plt.plot(accuracy_test,label='Testing Accuracy',color='green')

plt.legend(handles=[train,test])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
