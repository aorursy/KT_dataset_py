# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

data = pd.read_csv('../input/ex1data2.txt', header = None) #read from dataset

X = data.iloc[:,0:2]

y = data.iloc[:,2]

m = len(y)

X.head()

# Any results you write to the current directory are saved as output.
# plotting the data

# don't know how to plot yet
# Feature Normalization-

X = (X - np.mean(X))/(np.std(X))
# addinng intercept term

y = y[:,np.newaxis]

theta = np.zeros([3,1])

iterations = 1400

alpha = 0.01

ones = np.ones((m,1))

X = np.hstack((ones,X))

X.view()
# calculating the cost

def computeCost(X, y, theta):

    temp = np.dot(X, theta) - y

    return np.sum(np.power(temp,2))/(2*m)



J = computeCost(X,y,theta)

print(J)
# finding optimal parameters using gradient descent

def GradientDescent(X,y,theta,alpha,iterations,m):

    for i in range(iterations):

        temp = np.dot(X,theta) - y

        temp = np.dot(X.T, temp)

        theta = theta - (alpha/m)*temp

    return theta

theta = GradientDescent(X,y,theta,alpha,iterations,m)

print(theta)
J = computeCost(X,y,theta)

print(J)