%matplotlib inline

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

import numpy as np

import pandas as pd
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

plt.subplot(1,1,1)



x_train = [3, 4, 5.5, 6.7, 8, 9.5]

y_train= [1.139, 0.67, 1.222, 0.87, 1.041, 0.982]



x_test = [2, 3.5, 5, 6.5, 7.8, 9.2]

y_test = [0.91, 1.17, 0.82, 1.17, 0.84, 1.08]



plt.scatter(x_train, y_train)

plt.scatter(x_test, y_test, c='r')

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

plt.subplot(1,1,1)



x = np.linspace(2, 10, 1000)

plt.subplot(2,1,1)

def f1(x):

    return (1 + (np.sin(2*x))/(1-x))



plt.plot(x, f1(x))



x_train = [3, 4, 5.5, 6.7, 8, 9.5]

y_train= [1.139, 0.67, 1.222, 0.87, 1.041, 0.982]



x_test = [2, 3.5, 5, 6.5, 7.8, 9.2]

y_test = [0.91, 1.17, 0.82, 1.17, 0.84, 1.08]



plt.scatter(x_train, y_train)

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

#fig.subplots_adjust(hspace=0.1, wspace=0)



x = np.linspace(2, 10, 1000)

plt.subplot(2,1,1)

def f1(x):

    return (1 + (np.sin(2*x))/(1-x))



plt.plot(x, f1(x))



x_train = [3, 4, 5.5, 6.7, 8, 9.5]

y_train= [1.139, 0.67, 1.222, 0.87, 1.041, 0.982]



x_test = [2, 3.5, 5, 6.5, 7.8, 9.2]

y_test = [0.91, 1.17, 0.82, 1.17, 0.84, 1.08]



plt.scatter(x_train, y_train)

plt.scatter(x_test, y_test, c='r')

plt.tick_params(axis='y', labelleft=True)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

#fig.subplots_adjust(hspace=0.1, wspace=0)



x = np.linspace(2, 10, 1000)

plt.subplot(2,1,1)

def f1(x):

    return (1 + (np.sin(2*x))/(1-x))



plt.plot(x, f1(x))



x_train = np.array([3, 4, 5.5, 6.7, 8, 9.5]).reshape(-1, 1)

y_train= np.array([1.139, 0.67, 1.222, 0.87, 1.041, 0.982]).reshape(-1, 1)



x_test = np.array([2, 3.5, 5, 6.5, 7.8, 9.2]).reshape(-1, 1)

y_test = np.array([0.91, 1.17, 0.82, 1.17, 0.84, 1.08]).reshape(-1, 1)



plt.scatter(x_train, y_train)

plt.scatter(x_test, y_test, c='r')

plt.tick_params(axis='y', labelleft=True)



# Training the Algorithm (Linear Regression):

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(x_train, y_train)



# Making prediction

y_pred = regressor.predict(x_test)

plt.plot(x_test, y_pred, color='red')

print('Intercept: ', regressor.intercept_)

print('Solpe: ', regressor.coef_)



# Draw lines represents error:

# line for linear regression

x_reg = [3.5, 3.5]

y_reg = [0.98130822, 1.17]

plt.plot(x_reg, y_reg, c = "r")



# line for linear regression

x_curve = [6.5, 6.5]

y_curve = [1.17, f1(6.5)]

plt.plot(x_curve, y_curve, c='g')



plt.show()
# calculate residuals for curve function

x_test = [2, 3.5, 5, 6.5, 7.8, 9.2]

y_curve_test = [f1(x) for x in x_test]

y_curve_test =np.array(y_curve_test).reshape(-1,1)



curve_residual = sum((y_test - y_curve_test)**2)

print('ResidualError for training data using curve:', 'zero')

print('ResidualError for testing data using curve:', curve_residual)

print()



# calculate residuals for training data with regression line

# reg_line equation:

y_reg_train = 0.97324915 + 0.00230259 * x_train

reg_residual_train = sum((y_train - y_reg_train)**2)

print('ResidualError for training data using LR:', reg_residual_train)



# calculate residuals for test data with regression prediction 

reg_residual_test = sum((y_test - y_pred)**2)

print('ResidualError for testing data using LR:', reg_residual_test)


