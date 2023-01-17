# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("/kaggle/input/logistic-regression/Social_Network_Ads.csv")
data.drop(["User ID"],axis =1, inplace= True)
data = pd.get_dummies(data)
y = data.Purchased.values
x = data.drop(["Purchased"],axis=1)
x

# Normalization
x_data = (x-np.min(x))/(np.max(x)-np.min(x)).values
x_data
#split test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y, test_size=0.2, random_state=42)

x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T
# Initialize the parameters weights and bias

def init_params(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
# forward and backward propagation

def for_back_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -(1-y_train)*np.log(1-y_head)-y_train*np.log(y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    # backward
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]  
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients
    
    
# updating parameters
def update(w,b, x_train, y_train, learning_rate, num_it):
    cost_list = []
    cost_list2=[]
    index = []
    
    for i in range(num_it):
        cost, gradients = for_back_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w-learning_rate*gradients["derivative_weight"]
        b = b-learning_rate*gradients["derivative_bias"]
        if i %10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f"%(i,cost))
    
    parameters = {"weight":w,"bias":b}
    
    return parameters, gradients, cost_list
def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    y_pred= np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i]< 0.5 :
            y_pred[0,i] = 0
        else:
            y_pred[0,i] = 1
    return y_pred
    
def log_regression(x_train,y_train,x_test,y_test, learning_rate,num_it):
    dimension = x_train.shape[0]
    w,b = init_params(dimension)
    
    parameters, gradients, cost_list = update(w,b,x_train,y_train, learning_rate,num_it)
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))    
    
log_regression(x_train,y_train,x_test,y_test,learning_rate=1,num_it=400)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
