import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from sklearn.preprocessing import OneHotEncoder as ohe
from sklearn import preprocessing as pre

dt = pd.read_csv('../input/kcp1.csv')
dt.info()
dt.describe()
z = dt.dropna(axis = 0)
Y = z[['ID', 'Sex', 'Sport']]
X = z[['Age', 'Height', 'Weight', 'Season', 'Year', 'Games', 'Name', 'Team', 'NOC', 'City', 'Event', 'Medal']]

a = pre.LabelEncoder()
b = pre.LabelEncoder()

d = Y.apply(a.fit_transform)
e = X.apply(b.fit_transform)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print(sigmoid)
import matplotlib.pyplot as plt  
print(len(d))
print(len(e))

plt.plot(e, d)
plt.xlabel('Y')
plt.ylabel('X')
plt.show()
nr = np.arange(-10, 10)
fig, ax = plt.subplots(figsize = (12,8))
ax.plot(nr, sigmoid(nr), 'g')
def cost(theta, e, d, lr) :
    theta = np.matrix(theta) 
    e = np.matrix(e)
    d = np.matrix(d)
        
    fst = np.multiply(-e, np.log(sigmoid(d * theta.T)))
    scd = np.multiply((1 - e), np.log(1 - np.sigmoid(d * theta.T)))
    reg = (lr / 2 * len(e)) * np.sum(np.power(theta[:, 1:theta.shape[1]], 2))
    
    return np.sum(fst - scd) / (len(e)) + reg 
def gradient_with_loop(theta, e, d, lr):
    
    theta = np.matrix(theta)
    e = np.matrix(e)
    d = np.matrix(d)
    
    para = int(theta.ravel().shape[1])
    grad = np.zeros(para)
    
    err = sigmoid(e * theta.T) - d
    
    for i in range(para):
        term = np.multiply(err, e[:,i])
        
        if (i == 0):
            grad[i] = np.sum(term) / len(e)
        else:
            grad[i] = (np.sum(term) / len(e)) + ((lr / len(e)) * theta[:,i])
    
    return grad
def gradient(theta, e, d, lr):
    
    theta = np.matrix(theta)
    e = np.matrix(e)
    d = np.matrix(d)
    
    para = int(theta.ravel().shape[1])
    err = sigmoid(e * theta.T) - d
    
    grad = ((e.T * err) / len(e)).T + ((lr / len(e)) * theta)
    
    # intercept gradient is not regularized
    grad[0, 0] = np.sum(np.multiply(err, X[:,0])) / len(e)
    
    return np.array(grad).ravel()
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr

X_train, X_test, Y_train, Y_test = tts(e, d, test_size = 0.25, random_state = 0)
rg  = lr()
rg.fit(X_train, Y_train)

y_pred = rg.predict(X_test)
avg = np.mean(y_pred)

reg = rg.score(X_test, y_pred)
print(avg)
print(reg)
print(y_pred)
def one_vs_all(e, d, nl, lr):
    rw = e.shape[0]
    para = e.shape[1]
    
    # k X (n + 1) array for the parameters of each of the k classifiers
    ala = np.zeros((nl, para + 1))
    
    # insert a column of ones at the beginning for the intercept term
    e = np.insert(e, 0, values=np.ones(rw), axis = 1)
    
    # labels are 1-indexed instead of 0-indexed
    for i in range(1, nl + 1):
        theta = np.zeros(para + 1)
        d_i = np.array([1 if label == i else 0 for label in d])
        d_i = np.reshape(d_i, (rw, 1))
        
        # minimize the objective function
        fmin = minimize(fun = cost, x0 = theta, args = (e, y_i, lr), method = 'TNC', jac = gradient)
        ala[i - 1,:] = fmin.e
    
    return ala

print(one_vs_all)
def predict_all(e, ala) :
    
    rows = e.shape[0]
    para = e.shape[1]
    nl = ala.shape[0]
    
    # same as before, insert ones to match the shape
    e = np.insert(e, 0, values = np.ones(rows), axis = 1)
    
    # convert to matrices
    e = np.matrix(e)
    ala = np.matrix(ala)
    
    # compute the class probability for each class on each training instance
    h = sigmoid(e * all_theta.T)
    
    # create array of the index with the maximum probability
    h_argmax = np.argmax(h, axis = 1)
    
    # because our array was zero-indexed we need to add one for the true label prediction
    h_argmax = h_argmax + 1
    
    return h_argmax

print(predict_all)