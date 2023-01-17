# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
from numpy import newaxis, r_, c_, mat
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print('Loading data ...\n');

# Load Data
data = pd.read_csv('../input/ex1data2.txt', header = None) #read from dataset
data.head(5)
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]
m = len(y)
print('size of house, no. of bedrooms,  price')
for i in range(10):
     print('{:8.0f}{:10.0f}{:20.0f}'.format(X.iloc[i,0], X.iloc[i,1], y[i]))
def featureNormalize(X):
    """
    Normalizes the features in x.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features) = Features to be normalized.

    Returns
    -------
    X_norm : ndarray, shape (n_samples, n_features)
        A normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    mu : ndarray, shape (n_features,)
        The mean value.
    sigma : ndarray, shape (n_features,)
        The standard deviation.
    """
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])
    
    
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma
X_norm, mu, sigma = featureNormalize(X)
print('Computed mean:', mu)
print('Computed standard deviation:', sigma)


# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X_norm))
# Choose some alpha value
alpha = 0.15
num_iters = 400

# Init theta and run gradient descent
theta = np.zeros(3)
def h(theta,X): #Linear hypothesis function
    return np.dot(X,theta)
def computeCostMulti(X,y,theta): #Cost function
    """
    theta is an n- dimensional vector of initial theta guess
    X is matrix with n- columns and m- rows
    y is a matrix with m- rows and 1 column
    """
    #note to self: *.shape is (rows, columns)
    J = float((1./(2*m)) * np.dot((h(theta,X)-y).T,(h(theta,X)-y)))
    return J
print ('Calculate cost with Initial Theta value: ',computeCostMulti(X,y,theta))
def gradientDescentMulti(X, y, theta, alpha, iterations):
    """
    Parameters
    X = ndarray, shape (n_samples, n_features)(m*n) = Training Data
    y = ndarray, shape (n_samples) = Labels
    theta = ndarray , shape(n_features) = n. Theta is an Initial linear regression parameter
    alpha = float, it's a learning rate
    
    """
    m = len(y)
    J_history = np.zeros(iterations)
    for i in range(num_iters):
        theta -= alpha / m * ((h(theta,X) - y).T.dot(X))
        J_history[i] = computeCostMulti(X, y, theta)

    return theta, J_history
# perform linear regression on the data set
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)
# get the cost (error) of the model
print ('Further testing and calculate theta after performing gradient decent:',computeCostMulti(X,y,theta))
print('theta:', theta.ravel())

plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost_J')
plt.show()
normalize_data = (np.array([1650,3])-mu)/sigma
normalize_data = np.hstack((np.ones(1), normalize_data))
price = normalize_data.dot(theta)
print('Predicted price for a 1650 sq-ft, 3 br house:', price)
data = pd.read_csv('../input/ex1data2.txt', header = None) #read from dataset
X = data.iloc[:, 0:2]
y = data.iloc[:, 2]
m = len(y)

# Add intercept term to X
X = np.hstack((np.ones((m, 1)), X))
def normalEqn(X,y):
    pinv_X = np.linalg.pinv(np.dot(X.T,X))
    X_y = np.dot(X.T,y)
    theta = np.dot(pinv_X,X_y)
    return np.linalg.inv(X.T.dot( X )).dot( X.T ).dot( y )
theta3 = normalEqn(X,y)
print (theta3)
price = np.array([(np.ones(1)), 1650, 3]).dot(theta3)
print ('Predicted price for a 1650 sq-ft, 3 br house (using normal equations):', price)
