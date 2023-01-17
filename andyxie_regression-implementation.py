import numpy as np

import pandas as pd 

import matplotlib.pylab as plt

%matplotlib inline
X = np.random.randint(0, 100, size = (4,100))

W = np.array([0.3, 1, 0.7, 1.14])

C = 0

Y = W.T.dot(X) + C + np.random.randint(0,5,size=(100,))

Y[:5]
X[:4,:10]
from numpy.linalg import inv

xTx = inv(np.dot(X.T, X))

#inv().dot(X.T)

#inv(X.T.dot(X)).dot(X.T.dot(Y))

X.T.dot(Y)
Y.shape
xTx.shape
xTx.dot(X.T).dot(Y.T)
X.dot(Y.T)