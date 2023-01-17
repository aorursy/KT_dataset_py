import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split



import os

print(os.listdir("../input"))

dataset = pd.read_csv("../input/heart.csv")



a = pd.get_dummies(dataset['cp'], prefix = "cp")

b = pd.get_dummies(dataset['thal'], prefix = "thal")

c = pd.get_dummies(dataset['slope'], prefix = "slope")



frames = [dataset, a, b, c]

dataset = pd.concat(frames, axis = 1)

dataset = dataset.drop(columns = ['cp', 'thal', 'slope'])



y = dataset.target.values

X_data = dataset.drop(['target'], axis = 1)



X = (X_data - np.min(X_data))/(np.max(X_data) - np.min(X_data))



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train = X_train.T

X_test = X_test.T

y_train = y_train.T

y_test = y_test.T
def sigmoid(z):

    s = 1/(1 + np.exp(-z))

    return s
def initialize(dim):

    w = np.full((dim,1),0.01)

    b = 0

    return w,b
def propogate(X,y,w,b,l):

    m = X.shape[1]

    h = sigmoid(np.dot(w.T,X) + b)

    cost = -(np.dot(y,np.log(h).T) + np.dot(1-y,np.log(1-h).T) + l*np.sum(w**2)/2)/m

    dw = (np.dot(X,(h-y).T) + l*(w))/m

    db = np.sum(h-y)/m

    grads = {"dw": dw,

              "db": db}

    return cost,grads
def calculate(X,y,w,b,num_iteration,learning_rate,l):

    m = X.shape[1]

    dim = X.shape[0]

    costs = []

    for i in range(num_iteration):

        cost,grads = propogate(X,y,w,b,l)

        costs.append(cost)

        dw = grads["dw"]

        db = grads["db"]

        w = w - learning_rate*dw

        b = b - learning_rate*db

    params = {"w":w,

              "b":b}

    return params,costs,grads
def convert(X,w,b):

    m = X.shape[1]

    h = np.zeros((1,m))

    a = sigmoid(np.dot(w.T,X) + b)

    for i in range(a.shape[1]):

        if(a[:,i]>=0.5):

            h[:,i] = 1

        else:

            h[:,i] = 0

    return h
dim = X_train.shape[0]

w,b = initialize(dim)

params,costs,grads = calculate(X_train,y_train,w,b,100,0.1,50)

w = params["w"]

b = params["b"]

h = convert(X_test,w,b)

print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(h - y_test))*100)/100*100))