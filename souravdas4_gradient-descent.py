!pip install prettytable
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_boston
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from sklearn.linear_model import SGDRegressor
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from prettytable import PrettyTable
x = PrettyTable(["No", "Algo", "lr_rate_variation", "alpha", "weights", "MSE"])
X = load_boston().data
Y = load_boston().target
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)
clf = SGDRegressor(learning_rate='constant')
clf.fit(X, Y)
MSE=mean_squared_error(Y, clf.predict(X))
print(clf.coef_)
print(clf.intercept_)
print(clf.get_params)

x.add_row([1,"SGDRegressor","constant",clf.get_params()['alpha'],clf.coef_,MSE])
# Gradient Descent calculates the error for each example within the training dataset, 
# but only after all training examples have been evaluated, the model gets updated.

def gradient_descent(x,y,theta,b,alpha,iteration):
    m=x.shape[0]
    for i in range(iteration):
        y_pred=x.dot(theta)+b
        loss=y_pred-y
        cost=(1/2*float(m))*np.sum(np.square(loss))
        theta=theta-(2/float(m))*alpha*(x.T.dot(loss))
        b=b-(2/float(m))*alpha*sum(loss)
    return theta,b,cost    
(theta,b,cost)=gradient_descent(X,Y,[0,0,0,0,0,0,0,0,0,0,0,0,0],1,0.01,100)

print(theta)
print(b)
MSE=mean_squared_error(Y, X.dot(theta)+b)

x.add_row([2,"Manual gradient_descent","constant",0.01,theta,MSE])
# Mini-batch Gradient Descent splits the training dataset into small batches 
# and performs an update for each of these batches.

def mini_batch_gradient_descent(x,y,theta,b,alpha,iteration,batch):
    m=x.shape[0]
    for i in range(iteration):
        idx = np.random.randint(m, size=batch)
        x1=x[idx,:]
        y1=y[idx]
        m1=x1.shape[0]
        y_pred=x1.dot(theta)+b
        loss=y_pred-y1
        cost=(1/2*float(m1))*np.sum(np.square(loss))
        theta=theta-(2/float(m1))*alpha*(x1.T.dot(loss))
        b=b-(2/float(m1))*alpha*sum(loss)
    return theta,b,cost
(theta,b,cost)=mini_batch_gradient_descent(X,Y,[0,0,0,0,0,0,0,0,0,0,0,0,0],1,0.01,100,100)

print(theta)
print(b)
MSE=mean_squared_error(Y, X.dot(theta)+b)

x.add_row([3,"Manual mini_batch_gradient_descent","constant",0.01,theta,MSE])
# Stochastic gradient descent (SGD) updates the parameters for each training example within the dataset, one by one.

def stochastic_gradient_descent(x,y,theta,b,alpha,iteration):
    for i in range(iteration):
        for row,target in zip(x,y):
            row=row.reshape(1,13)
            m=row.shape[0]
            y_pred=row.dot(theta)+b
            loss=y_pred-target
            theta=theta-(2/float(m))*alpha*(row.T.dot(loss))
            b=b-(2/float(m))*alpha*sum(loss)
    return theta,b,cost
(theta,b,cost)=stochastic_gradient_descent(X,Y,[0,0,0,0,0,0,0,0,0,0,0,0,0],1,0.0001,100)

print(theta)
print(b)
MSE=mean_squared_error(Y, X.dot(theta)+b)

x.add_row([4,"Manual stochastic_gradient_descent","constant",0.0001,theta,MSE])
print(x)