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
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
data = pd.read_csv("/kaggle/input/week-1-ml-andrew-ng/ex1data1.txt", delimiter = ',')
data.head()
data.describe()
X = data.iloc[:, 0]

y = data.iloc[:, 1]

print(X)

print(y)
data.columns = ['Population', 'Profit']

data.head()
plt.scatter(X, y, color = 'red')

plt.xlabel("Population of City (10,000s)")

plt.ylabel("Profit ($10,000")

plt.title("Profit Vs Population")
X = X[:,np.newaxis]

y = y[:,np.newaxis]

theta = np.zeros([2,1])

iterations = 1500

alpha = 0.01

m = len(y)

ones = np.ones((m,1))

X = np.hstack((ones, X)) # adding the intercept term
print(X)
def computeCost(X, y, theta):

    

    m = len(y)

    hypothesis_function = X.dot(theta)

    squared_error = (hypothesis_function - y) ** 2

    cost_function = float(1/ (2 * m) * np.sum(squared_error))

    return cost_function

print(computeCost(X,y,theta))
computeCost(X, y, theta)
def gradientDescent(X, y, theta, alpha, iterations):

    for i in range(iterations): #updating theta till convergence

        hypothesis_function = np.dot(X, theta)

        error = np.dot(X, theta) - y

        cost_function_derivative = np.dot(X.T, error)

        theta = theta - (alpha/m) * cost_function_derivative #updating theta

    return theta

theta = gradientDescent(X, y, theta, alpha, iterations)

print(theta)
X[:, 0]
X1 = X[:, 1]

plt.scatter(X1, y)

plt.xlabel('Population of City in 10,000s')

plt.ylabel('Profit in $10,000s')

hypothesis_function = np.dot(X, theta)

plt.plot(X1, hypothesis_function)

plt.show()