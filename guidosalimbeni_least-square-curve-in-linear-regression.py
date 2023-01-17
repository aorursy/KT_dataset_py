import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from numpy.linalg import inv

import statsmodels.api as sm

import pylab as pl

import seaborn as sns

import matplotlib.pyplot as plt
x = np.arange(0,20)

# generate random noise (and the true y)

y = x + 10*np.random.random((1,20)) - 1
plt.scatter(x,y)
# calculate regression paramters

A = np.array([ x, np.ones(20)])

w = np.linalg.lstsq(A.T, y.T)[0] 

# plotting the line

line = w[0]*x + w[1]

pl.plot(line, 'r-', y.T, 'o')

pl.title('Linear Regression')

pl.xlabel('x')

pl.ylabel('y')

pl.show()
# coeff

w
X = x.reshape(-1, 1)

Y = y.reshape(20,1)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X,Y)

print (linear_regression.coef_, linear_regression.intercept_)


w_fake = np.array([[3],[4.0746206 ]])

# plotting the line

y_pred = w_fake[0]*x + w_fake[1]

pl.plot(y_pred, 'r-', y.T, 'o')

pl.title('Linear Regression')

pl.xlabel('x')

pl.ylabel('y')

pl.show()
error = np.sum (y - y_pred)**2

print (error)
listOfWeights = np.linspace(0.5, 3.0, num=100) # generating a selection of 100 possible parameters
# collecting the error for each of the selection of paramenters

errorList = []



for weight in listOfWeights:

    y_pred_temp = weight*x + w[1]

    error_temp = np.sum (y - y_pred_temp)**2

    errorList.append(error_temp)

    

errorArray   = np.array(errorList)

plt.scatter(listOfWeights, errorArray)