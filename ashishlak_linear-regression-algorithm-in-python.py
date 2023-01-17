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
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
# make data
# divide it into training and testing
# algorithm
    # 1. take random theta[theta0,theta1]
    # 2. cost function
    # 3. hypothesis
    # 4. derivative
    # 5. gradient descent
# find accuracy
# Data Prepare
X,y,coef=make_regression(n_samples=500,n_features=1,bias=4.2,noise=17.1,coef=True)
print(X.shape)
print(y.shape)
print(coef)
# plt.scatter(X,y)
# plt.show()
plt.scatter(X[:,0],y)
# r = 10
# plt.xlim(-r,r)
# plt.ylim(-r,r)
plt.show()
# divide it into training and testing data

split = int(0.8 * X.shape[0])
train_x = X[:split,0]
train_y = y[:split]

test_x = X[split:,0]
test_y = y[split:]

print(train_x.shape)
print(test_x.shape)
loss=[]
theta = [0.1,0.1]
def hypothesis(theta,train_x):
    return theta[0] + train_x * theta[1]

def cost_function(theta,train_x,train_y):
    
    err = 0.0
    for i in range(train_x.shape[0]):
        err += (hypothesis(theta,train_x[i]) - train_y[i])**2
    
    return err/2*train_x.shape[0] 
    
def min_cost_function(theta,train_x,train_y):
    
    grad_0 = 0.0
    grad_1 = 0.0
    
    for i in range(train_x.shape[0]):
        grad_0 += train_y[i] - hypothesis(theta,train_x[i]) * (-1)
        grad_1 += (train_y[i] - hypothesis(theta,train_x[i])) * (-1*train_x[i])
    
    grad_0 /= train_x.shape[0]
    grad_1 /= train_x.shape[0]
    
    return [grad_0,grad_1]

def grad_descent(theta,train_x,train_y,alpha=0.01):
    err = cost_function(theta,train_x,train_y)
    get_slope = min_cost_function(theta,train_x,train_y)
    
    theta[0] = theta[0] - alpha * get_slope[0]
    theta[1] = theta[1] - alpha * get_slope[1]
    
    return err,theta
for i in range(1001):
    l,theta = grad_descent(theta,train_x,train_y)
    loss.append(l)
    if i % 100 == 0:
        y_0 = hypothesis(theta, -3)
        y_1 = hypothesis(theta, 3)
        plt.scatter(X[:,0],y)
        plt.plot([-3, 3], [y_0, y_1], 'r*-')
        plt.show()
        
plt.plot(loss)
plt.show()
