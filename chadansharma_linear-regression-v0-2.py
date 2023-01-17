# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# https://mubaris.com/posts/linear-regression/



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (20.0, 10.0)

from mpl_toolkits.mplot3d import Axes3D



data = pd.read_csv('../input/student-linear/student.csv')

print(data.shape)

data.head()
math = data['Math'].values

read = data['Reading'].values

write = data['Writing'].values



# Ploting the scores as scatter plot

fig = plt.figure(figsize=(20,10))

ax = Axes3D(fig)

ax.scatter(math, read, write, color='#ef1234')

plt.show()
m = len(math)

x0 = np.ones(m)

X = np.array([x0, math, read]).T

# Initial Coefficients

B = np.array([0, 0, 0])

Y = np.array(write)

alpha = 0.0001
def cost_function(X, Y, B):

    m = len(Y)

    J = np.sum((X.dot(B) - Y) ** 2)/(2*m)

    return J
inital_cost = cost_function(X, Y, B)

print(inital_cost)
def gradient_descent(X, Y, B, alpha, iterations):

    cost_history = [0] * iterations

    m = len(Y)

    

    for iteration in range(iterations):

        # Hypothesis Values

        h = X.dot(B)

        # Difference b/w Hypothesis and Actual Y

        loss = h - Y

        # Gradient Calculation

        gradient = X.T.dot(loss) / m

#         print (gradient[0])

        # Changing Values of B using Gradient

        B = B - alpha * gradient

        # New Cost Value

        cost = cost_function(X, Y, B)

        cost_history[iteration] = cost

        

    return B, cost_history
# 100000 Iterations

newB, cost_history = gradient_descent(X, Y, B, alpha, 200000)



# New Values of B

print(newB)



# Final Cost of new B

print(cost_history[-1])
# Model Evaluation - RMSE

def rmse(Y, Y_pred):

    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))

    return rmse



# Model Evaluation - R2 Score

def r2_score(Y, Y_pred):

    mean_y = np.mean(Y)

    ss_tot = sum((Y - mean_y) ** 2)

    ss_res = sum((Y - Y_pred) ** 2)

    r2 = 1 - (ss_res / ss_tot)

    return r2



Y_pred = X.dot(newB)



print(rmse(Y, Y_pred))

print(r2_score(Y, Y_pred))
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error



# X and Y Values

X = np.array([math, read]).T

Y = np.array(write)



# Model Intialization

reg = LinearRegression()

# Data Fitting

reg = reg.fit(X, Y)

# Y Prediction

Y_pred = reg.predict(X)



# Model Evaluation

rmse = np.sqrt(mean_squared_error(Y, Y_pred))

r2 = reg.score(X, Y)



print(rmse)

print(r2)
cost_history = [0] * 100000

m = len(Y)
# for iteration in range(iterations):

#     # Hypothesis Values

h = X.dot(B)

# Difference b/w Hypothesis and Actual Y

loss = h - Y

# Gradient Calculation

gradient = X.T.dot(loss) / m

#         print (gradient[0])

B = B - alpha * gradient

cost_function(X, Y, B)
gradient
B
B
B