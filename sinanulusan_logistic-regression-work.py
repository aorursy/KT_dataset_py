# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/genderheightweightcsv/weight-height.csv")
data
data.Gender=[1 if each=='Male' else 0 for each in data.Gender]
y = data.Gender.values
x = data.drop(["Gender"],axis=1)
y
x = (x-np.min(x))/(np.max(x)-np.min(x)).values
x
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

def initialize_weights_and_bias(dimension):
    w = np.full((dimension,1),0.01)
    b=0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
def forward_backward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -(y_train*np.log(y_head)+(1-y_train)*np.log(1-y_head))
    cost = np.sum(loss)/(x_train.shape[1])
    
    derivative_weight = (np.dot(x_train,(y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    grad = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,grad
def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    cost_list=[]
    index=[]
    for i in range(number_of_iteration):
        cost , gradients = forward_backward_propagation(w,b,x_train,y_train)
        w = w - learning_rate*gradients["derivative_weight"]    
        b = b - learning_rate*gradients["derivative_bias"]
        if i % 100 == 0:
            cost_list.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    parameters={"weight":w,"bias":b}
    plt.plot(index,cost_list)
    plt.xticks(index,rotation='vertical')
    plt.xlabel("Number of iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters,gradients,cost_list
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    prediction = np.zeros((1,x_test.shape[1]))
    for i in range(z.shape[1]):
        if(z[0,i]<=0.5):
            prediction[0,i]=0
        else:
            prediction[0,i]=1
    return prediction
def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):
    w,b = initialize_weights_and_bias(x_train.shape[0])
    parameters,gradients,cost_list = update(w,b,x_train,y_train,learning_rate,number_of_iteration)
    
    prediction = predict(parameters["weight"],parameters["bias"],x_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(prediction - y_test)) * 100))
logistic_regression(x_train,y_train,x_test,y_test,learning_rate=0.3,number_of_iteration=3000)