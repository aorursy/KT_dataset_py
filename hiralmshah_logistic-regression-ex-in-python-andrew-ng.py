# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/ex2data1.txt')
data.head(5)
X = np.array(data.iloc[:, 0:2]) #for plotting data we need array instead of dataframe slice
y = np.array(data.iloc[:, 2]) #for plotting data we need array instead of series slice
#type(y)
def PlotData(X,y):
    """
    Plots the data points X and y.
    n_samples = 99 can varify from (X.shape, y.shape)
    """
    pos = np.argwhere(y == 1)#it will return np.ndarray instead of tuple if we use where.
    neg = np.argwhere(y == 0)#it will return np.ndarray instead of tuple if we use where.
  
    plt.plot(X[pos, 0], X[pos, 1], linestyle='', marker='+', color='k')
    plt.plot(X[neg, 0], X[neg, 1], linestyle='', marker='o', color='y')
plt.figure()
PlotData(X, y)
# Labels and Legend
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted', 'Not admitted'], loc='upper right', numpoints=1)
plt.show()
#Initialze training parameters
m, n =  X.shape

# Add intercept term
X = np.hstack((np.ones((m, 1)), X))

# Initialize fitting parameters
theta = np.zeros(n + 1)  
#Compute sigmoid function.
def sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g   

#evaluate sigmoid(0), it should give you 0.5
#sigmoid(0)
def h(theta,X):
    return sigmoid(np.dot(X,theta))
def costFunction(X,y,theta):
    J = (1/m) * np.sum((-y * np.log(h(theta,X))) - ((1-y)*np.log(1-h(theta,X))))
    
    #grad = (1/m) * ((h(theta,X)) - y)*X.T
    grad = (1/m) * (h(theta,X) - y).T.dot(X)
    
    return (J, grad)
cost, grad = costFunction(X,y,theta)
print('cost for intial theta value', cost)
print('gradient for initial theta value ', grad)
print(data.shape, X.shape, y.shape, theta.shape)
y = y.reshape(m,1)
theta = np.zeros((n+1,1))

print(data.shape, X.shape, y.shape, theta.shape)
import scipy.optimize as opt

theta, nfeval, rc = opt.fmin_tnc(func=costFunction, x0=theta, args=(X, y))

cost, grad = costFunction(X,y,theta)
print('cost for intial theta value', cost)
print('gradient for initial theta value ', grad)
