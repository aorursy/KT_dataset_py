# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np 
import pandas as pd 
from  sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



import os
print(os.listdir("../input"))

df = pd.read_csv("../input/voice.csv")
df.label = [1 if each == "male" else 0 for each in df.label]
print(df.info())

y = df.label.values
x_data = df.drop(["label"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b
def sigmoid(z):
    return 1/(1+np.exp(-z))
def forward_backward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost,gradients


def update(w,b,x_train,y_train,learning_rate,num_iteration):
    cost_list=[]
    cost_list2=[]
    index=[]
    
    for i in range(num_iteration):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    parameters={"weight":w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters,gradients,cost_list

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iterarion):
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
            
    parameters = {"weight": w,"bias": b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show()
    
    return parameters, gradients, cost_list     
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_pre = np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i]<=0.5:
            Y_pre[0,i] = 0
        else:
            Y_pre[0,i]=1
    
    return Y_pre

def logistic_regression(x_train,x_test,y_train,y_test,learning_rate,num_iteration):
    
    dimension = x_train.shape[0]
    w,b = initialize_weights_and_bias(dimension)
    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,num_iteration)
    y_pre_test = predict(parameters["weight"],parameters["bias"],x_test)
    print("test accuracy:{}%".format(100-np.mean(np.abs(y_pre_test-y_test)*100)))
    
    
logistic_regression(x_train,x_test,y_train,y_test,learning_rate=1,num_iteration=501)