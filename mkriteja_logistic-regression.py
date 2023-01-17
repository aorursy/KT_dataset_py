# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data = pd.read_csv('../input/Social_Network_Ads.csv')
data.drop(columns=['User ID','Gender',],axis=1,inplace=True)
data.head()
y = data.iloc[:,-1].values
X = data.iloc[:,:-1].values
# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
# Sacaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#reshape
X_train = X_train.T
X_test = X_test.T

y_train = y_train.reshape(1, -1)
y_test = y_test.reshape(1, -1)
def sigmoid(z):
    return 1/(1+np.exp(-z))
def initialize_with_zeros(dim):
    w = np.zeros((dim,1))
    b = 0
    
    return w,b
def propagate(w,b,X,Y):
    m = X.shape[1]
    
    A = sigmoid(np.dot(w.T,X)+b)
    cost = np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) / -m
    
    dw = np.dot(X,(A-Y).T)/m
    db = np.sum(A-Y)/m
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost
def optimize(w, b, X, Y, num_iterations, learning_rate):
     for i in range(num_iterations):
        grads, cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]
        
        w = w-(learning_rate*dw)
        b = b-(learning_rate*db)
        
        params = {"w": w,
              "b": b}
    
        grads = {"dw": dw,
             "db": db}
    
        return params, grads
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    Y_prediction = np.where(A > 0.5,1.,0.)
    
    return Y_prediction
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5):
    
    w,b = initialize_with_zeros(X_train.shape[0])
    
    parameters, grads = optimize(w,b,X_train,Y_train, num_iterations, learning_rate)
    
    w = parameters["w"]
    b = parameters["b"]
    
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
model(X_train, y_train, X_test, y_test, num_iterations = 50, learning_rate = 0.005)