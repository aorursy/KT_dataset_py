# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))

data = pd.read_csv('../input/ex1data1.txt', header = None) #read from dataset

X = data.iloc[:,0]

y = data.iloc[:,1]

m = len(y)

data.head()

# Any results you write to the current directory are saved as output.
# plotting the data

import matplotlib.pyplot as plt

plt.scatter(X,y)

plt.xlabel('popultaion in 10,000s')

plt.ylabel('profit in $10,000s')

plt.show()
# addinng intercept term

X = X[:,np.newaxis]

y = y[:,np.newaxis]

theta = np.zeros([2,1])

iterations = 1500

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
theta.shape
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
# plot showing the best fit line

plt.scatter(X[:,1],y)

plt.xlabel('Population of City in 10,000s')

plt.ylabel('Profit in $10,000s')

plt.plot(X[:,1], np.dot(X,theta))

plt.show()
# predict profit for population

def predict(x):

    return x*theta[1]+theta[0]



population = 6.3261

profit = predict(population)

print(profit)