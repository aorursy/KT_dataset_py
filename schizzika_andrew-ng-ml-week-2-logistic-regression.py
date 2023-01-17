# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/week-1-ml-andrew-ng/ex1data2.txt", delimiter = ',')

data.head()

X = data.iloc[:, 0:2] #independent variables

y = data.iloc[:, 2] #dependent variables
m = len(y)

#Normalisation

X = (X - np.mean(X))/ np.std(X)
ones = np.ones((m,1))

X = np.hstack((ones, X))

alpha = 0.01

num_iters = 400

theta = np.zeros((3,1))

y = y[:,np.newaxis]
def computeCostMulti(X, y, theta):

    temp = np.dot(X, theta) - y

    return np.sum(np.power(temp, 2)) / (2*m)

J = computeCostMulti(X, y, theta)

print(J)
def gradientDescentMulti(X, y, theta, alpha, iterations):

    m = len(y)

    for _ in range(iterations):

        temp = np.dot(X, theta) - y

        temp = np.dot(X.T, temp)

        theta = theta - (alpha/m) * temp

    return theta

theta = gradientDescentMulti(X, y, theta, alpha, num_iters)

print(theta)
J = computeCostMulti(X, y, theta)

print(J)