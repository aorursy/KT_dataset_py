# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.optimize as opt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

print(os.listdir("../input/"))

data = pd.read_csv('../input/ex2data1.txt', header = None) #read from dataset

X = data.iloc[:,0:2]

y = data.iloc[:,2]

m = len(y)

y.head()

# Any results you write to the current directory are saved as output.
# adding intercept term

y = y[:,np.newaxis]

(a,b) = X.shape

theta = np.zeros([b+1,1])

ones = np.ones((m,1))

X = np.hstack((ones,X))
# sigmoid Function

def sigmoid(x):

    return 1/(1+np.exp(-x))
# cost function

def cost(theta, X, y):

         J = (-1/m)*np.sum(np.multiply(y,np.log(sigmoid(X @ theta)))

                       + (np.multiply(1-y,np.log(1- sigmoid(X @ theta)))))

         return J



print(cost(theta, X, y))
# Gradient funtion

def gradient(theta, X, y):

    return ((1/m) * (np.dot(X.T , (sigmoid(X @ theta) - y))))
temp = opt.fmin_tnc(func = cost, 

                    x0 = theta.flatten(),fprime = gradient, 

                    args = (X, y.flatten()))

theta_optimized = temp[0]

print(theta_optimized)
J = cost(theta_optimized[:,np.newaxis], X, y)

print(J)