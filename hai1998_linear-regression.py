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
import matplotlib.pyplot as plt

import math
ftPath = "/kaggle/input/housingprice/ex1data2.txt"

lines = [] # Living area & number of bedroom

with open(ftPath) as f:

    for line in f:

        a = [float(i) for i in line.strip().split(',')]

        lines.append(a)

data = np.array(lines)
data
X = data[:,[0, 1]]

y = np.c_[data[:,2]]
print(len(X), len(y))
a = X[:, 0]

print(a)
plt.figure(figsize=(10, 5))

plt.plot(a, y, "bo")

# plt.axis([140, 190, 45, 75])

plt.xlabel("Living area")

plt.ylabel("Price")

xLine = np.array([700, 4500])

yLine = 50000 + 60*xLine

yOptima = 71270 + 134.5*xLine

yTest = 4 + 165*xLine

ySGD = -78.34792489 + 361.51173555*xLine

yMini = 165*xLine # similar to yTest

 

plt.plot(xLine, yLine, marker="o", color="r")

plt.plot(xLine, yOptima, marker="^", color="g")

plt.plot(xLine, yTest, marker="x", color="darkorange")

plt.plot(xLine, ySGD, color="lightblue")

plt.grid(True)

plt.show()
m = len(a)

A = 1/(2*m) * (a.T.dot(a))

B = -1/(2*m) * (a.T.dot(y))

C = 1/(2*m) * (y.T.dot(y))

print(A, B, C)

# 2310419.212765957 -382104564.09574467 65591548106.45744
plt.figure(figsize=(10, 5))

theta = np.linspace(-230, 400, 1000)

J = A*theta**2 + B*theta + C

plt.plot(theta, J.reshape(-1, 1), color='r')

plt.xlabel("Theta")

plt.ylabel("Cost function")

plt.grid(True)

plt.show()
# Solve the problem for theta0, theta1

from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()

lin_reg.fit(X, y)

print(lin_reg.intercept_, lin_reg.coef_)
from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()

lin_reg.fit(X, y)

print(lin_reg.intercept_, lin_reg.coef_)
theta_best = np.array([[89597.9], [139.21067402], [-8738.01911233]])

Xb = np.c_[np.ones((m, 1)), X]

error = 1/m * math.sqrt((Xb.dot(theta_best)-y).T.dot(Xb.dot(theta_best)-y))

print("MSE:", error)
m = len(X)
c = len(Xb[0])

theta = np.random.randn(c,1)

print(theta)

eta = 2e-7
# Each iteration

tmp1 = Xb.dot(theta) - y

gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

theta = theta - eta * gradients

err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

print("MSE:", err) 

print(theta)
# TODO: Add early stopping criteria if err doesn't decrease more than epsilon value

# TODO: Check the number of iteration to converge

epsilon = 2e-5

for i in range(1000):

    tmp1 = Xb.dot(theta) - y

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    oldErr = err

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    if abs(err-oldErr) < epsilon:

        break

print('err', err)

print('theta:', theta)

print("MSE:", err)

print("abs err:", abs(err-oldErr))

print("-"*10)
theta = np.array([[90000], [120], [-8000]])

print(theta)

eta = 2e-7
tmp1 = Xb.dot(theta) - y

gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

theta = theta - eta * gradients

err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

print("MSE:", err)

print(theta)
# TODO: Add early stopping criteria if err doesn't decrease more than epsilon value

# TODO: Check the number of iteration to converge

epsilon = 2e-6

for i in range(1000):

    tmp1 = Xb.dot(theta) - y

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    oldErr = err

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    if abs(err-oldErr) < epsilon:

        break

print('gradients:', gradients)

print('theta:', theta)

print("MSE:", err)

print("abs err:", abs(err-oldErr))

print("-"*10)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(Xb)

scaleX = scaler.transform(Xb)
c = len(Xb[0])

theta = np.random.randn(c,1)

print(theta)

eta = 0.5
for i in range(1000):

    tmp1 = scaleX.dot(theta) - y

    gradients = (2/m) * scaleX.T.dot(scaleX.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((scaleX.dot(theta)-y).T.dot(scaleX.dot(theta)-y))

print(i)

print(theta)

print("MSE:", err)

print("-"*10)
# TODO Question: How to convert theta from scaled value back to normal value? Derive the equation.
nEpochs = 60

t0, t1 = 2, 10000000 # learning schedule hyperparameters



def learningSchedule(t): 

    return t0/(t+t1)



c = len(Xb[0])

theta = np.random.randn(c,1) # random initialization

print(theta)

for epoch in range(nEpochs): 

    for i in range(m):

        # TODO: Implement

        # - randomly take 1 sample

        rand_int = np.random.randint(0, m)

        xb_i = Xb[rand_int,:].reshape(1, Xb.shape[1])

        y_i  = y[rand_int].reshape(1,1)

        # - update gradients value by that sample

        eta = learningSchedule(epoch*m + i)

        theta = theta - eta * gradients

        err = 1/m * math.sqrt((xb_i.dot(theta)-y_i).T.dot(xb_i.dot(theta)-y_i))

#         print("Iteration:", epoch*m + i)

#         print("Eta:", eta)

print(theta)

print("MSE:", err)

print("-"*10)
nEpochs = 100

batchSize = 10

t0, t1 = 2, 10000000 # learning schedule hyperparameters



def learningSchedule(t): 

    return t0/(t+t1)

c = len(Xb[0])

theta = np.random.randn(c,1) # random initialization

print(theta)

batchSize = 10

for epoch in range(nEpochs): 

    # TODO: Implement

    # - randomly take 10 samples (batchSize)

    indices = np.random.permutation(m)

    X = Xb[indices]

    Y = y[indices]

    # - update gradients value by those samples

    # - change the batchSize value to see the difference

    for i in range(0,m,batchSize):

        xb_i = X[i:i+batchSize]

        y_i  = Y[i:i+batchSize]

        xb_i = np.c_[np.ones(len(xb_i)),xb_i]

        

        eta = learningSchedule(epoch)

        theta = theta - eta * gradients

        err = 1/m * math.sqrt((xb_i.dot(theta)-y_i).T.dot(xb_i.dot(theta)-y_i))

#         print(epoch)

#         print("Eta:", eta)

print(theta)

print("MSE:", err)