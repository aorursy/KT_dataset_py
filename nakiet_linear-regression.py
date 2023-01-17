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
import math

from sklearn.linear_model import LinearRegression
ftPath = "/kaggle/input/housingprice/ex1data2.txt"

lines = [] # Living area & number of bedroom

with open(ftPath) as f:

    for line in f:

        a = [float(i) for i in line.strip().split(',')]

        lines.append(a)

data = np.array(lines)
# Input: Living area & number of bedrooms

# Output: Predicted price

_input = data[:, [0, 1]]

input_len = len(_input) # Number of training examples (47)



# y = theta0 * x0 + theta1 * x1 + theta2 * x2

# theta0 * x0: intercept

# x0 always equal to 1

x0 = np.ones((input_len, 1)) # matrix (47 x 1)

# np.c_: Translates slice objects to concatenation along the second axis.

# Training data point x_i

# x[][0]: x0

# x[][1]: x1

# x[][2]: x2

x = np.c_[x0, _input] # matrix (47 x 3)

# Observed price

y = data[:, [2]]



# Number of thetas (3)

theta_len = len(x[0])

# np.random.randn: Creates an array of specified shape (3, 1)

# and fills it with random values as per standard normal distribution.

theta = np.random.randn(theta_len, 1)



# learning rate

eta = 2e-7
lin_reg = LinearRegression()

lin_reg.fit(_input, y)

print(lin_reg.intercept_, lin_reg.coef_)



best_theta = np.array([[89597.9095428], [139.21067402], [-8738.01911233]])

best_error = (1 / 3) * math.sqrt(best_theta.T.dot(best_theta))

print(best_error)
def getThetaAndError(_x, _theta, _y, _eta):

    x_len = len(_x)

    # Residual array

    residuals = _x.dot(_theta) - _y

    # Sum of the squared residuals

    #             [[1]

    # [1, 2, 3] *  [2]  = 1^2 + 2^2 + 3^2 = 14

    #              [3]]

    gradient_descent = (2 / x_len) * _x.T.dot(residuals)

    # New theta

    _theta = _theta - _eta * gradient_descent

    # New residual list

    new_residuals = _x.dot(_theta) - _y

    # Error

    _error = (1 / x_len) * math.sqrt(new_residuals.T.dot(new_residuals))

    

    return _theta, _error
res = getThetaAndError(x, theta, y, eta)

theta = res[0]

new_error = res[1]

old_error = new_error * 2

i = 0



while old_error - new_error > 0.0005:

    res = getThetaAndError(x, theta, y, eta)

    theta = res[0]

    old_error = new_error

    new_error = res[1]

    i = i + 1

    print(i)

    print(theta)

    print(new_error)

    print(old_error - new_error)
def getThetaAndCostFunction(_x, _theta, _y, _eta):

    x_len = len(_x)

    # Residual array

    residuals = _x.dot(_theta) - _y

    # Sum of the squared residuals

    #             [[1]

    # [1, 2, 3] *  [2]  = 1^2 + 2^2 + 3^2 = 14

    #              [3]]

    gradient_descent = (2 / x_len) * _x.T.dot(residuals)

    # New theta

    _theta = _theta - _eta * gradient_descent

    # New residual list

    new_residuals = _x.dot(_theta) - _y

    # Cost function

    _cost_function = (1 / x_len) * new_residuals.T.dot(new_residuals)

    

    return _theta, _cost_function



res = getThetaAndCostFunction(x, theta, y, eta)

theta = res[0]

new_cost_function = res[1]

old_cost_function = new_cost_function * 2

i = 0



while old_cost_function - new_cost_function > 294:

    res = getThetaAndCostFunction(x, theta, y, eta)

    theta = res[0]

    old_cost_function = new_cost_function

    new_cost_function = res[1]

    i = i + 1

    print(i)

    print(theta)

    print(old_cost_function - new_cost_function)
nEpochs = 60

t0, t1 = 2, 10000000 # learning schedule hyperparameters



def learningSchedule(t): 

    return t0/(t+t1)



m = input_len

for epoch in range(nEpochs): 

    # Array contains chosen index

    index_array = []

    

    for i in range(m):

        # TODO: Implement

        # - randomly take 1 sample

        # Random training sample index (0 <= index < m = 47)

        index = np.random.randint(0, m)

        # Loop when index was chosen

        while index in index_array:

            index = np.random.randint(0, m)

        index_array.append(index)

        

        # numpy.reshape: Gives a new shape to an array without changing its data.

        # (3,) -> (1, 3)

        x_i = x[index,:].reshape(1, theta_len)

        # (1,) -> (1, 1)

        y_i = y[index].reshape(1, 1)

        

        # - update gradients value by that sample

        eta = learningSchedule(epoch*m + i)

        res = getThetaAndError(x_i, theta, y_i, eta)

        theta = res[0]

        error = res[1]

        print("Iteration:", epoch*m + i)

        print("Eta:", eta)

        print(theta)

        print("MSE:", error)

        print("-"*10)
nEpochs = 100

batchSize = 10

t0, t1 = 2, 10000000 # learning schedule hyperparameters



def learningSchedule(t): 

    return t0/(t+t1)



m = input_len

for epoch in range(nEpochs): 

    # Array contains chosen index

    index_array = []

    

    # TODO: Implement

    remaining_size = m - len(index_array)

    while remaining_size > 0:

        x_batch = []

        y_batch = []

        

        # - randomly take 10 samples (batchSize)

        if remaining_size < batchSize:

            batchSize = remaining_size

                

        for i in range(batchSize):

            # Random training sample index (0 <= index < m = 47)

            index = np.random.randint(0, m)

            # Loop when index was chosen

            while index in index_array:

                index = np.random.randint(0, m)

                

            index_array.append(index)

            x_batch.append(x[index,:])

            y_batch.append(y[index])

            remaining_size = m - len(index_array)



        # - update gradients value by those samples

        eta = learningSchedule(epoch)

        res = getThetaAndError(np.array(x_batch), theta, np.array(y_batch), eta)

        theta = res[0]

        error = res[1]

        print(epoch)

        print("Eta:", eta)

        print(theta)

        print("MSE:", error)

        

        # - change the batchSize value to see the difference