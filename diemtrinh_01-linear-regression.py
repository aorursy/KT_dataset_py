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
X = data[:,[0, 1]] # get all rows, col 0: Living Area, 1: Bedrooms

y = np.c_[data[:,2]] # get all rows, col 2: Price
print(len(X), len(y))
a = X[:, 0] # get all rows, col 0: Living Area

print(a)
plt.figure(figsize=(10, 5))

plt.plot(a, y, "bo") # ve duong thang y = ax + b

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
m = len(X)

# Translates slice objects to concatenation along the second axis.

# add slice 1..1 to X

Xb = np.c_[np.ones((m, 1)), X]
theta_best = np.array([[89597.9], [139.21067402], [-8738.01911233]])

# .T: chuyen vi ma tran: chuyen dong thanh cot

# .dot: nhan ma tran

error = 1/m * math.sqrt((Xb.dot(theta_best)-y).T.dot(Xb.dot(theta_best)-y))

print("MSE:", error)
m = len(X)

Xb = np.c_[np.ones((m, 1)), X] # create new matrix with m rows, 1 col and merge with matrix X

# print('Xb:', Xb)

# print('X:', X)
c = len(Xb[0]) # count number col of row Xb[0]

theta = np.random.randn(c,1) # random the matrix with c rows, 1 col

print(theta)

eta = 2e-7
# Each iteration

tmp1 = Xb.dot(theta) - y

gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

theta = theta - eta * gradients # B0 = trung bình (Y) - B1 * Trung bình (X)

err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y)) # cong thuc tinh sai so e

print("MSE:", err) 

print(theta)
# TODO: Add early stopping criteria if err doesn't decrease more than epsilon value

# TODO: Check the number of iteration to converge

epsilon = 0.0005

decreaseError = epsilon + 1

for i in range(1000):

    tmp1 = Xb.dot(theta) - y

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    print(i)

    print(theta)

    print("MSE:", err)

    print("-"*10)

    if (i != 0): decreaseError = prevErr - err

    prevErr = err

    if decreaseError <= epsilon:

        print('i:', i)

        break
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

epsilon = 0.0005

decreaseError = eta + 1

for i in range(1000):

    tmp1 = Xb.dot(theta) - y

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    print(i)

    print(gradients)

    print(theta)

    print("MSE:", err)

    print("-"*10)

    if (i != 0): decreaseError = prevErr - err

    prevErr = err

    if decreaseError <= epsilon:

        print('i:', i)

        break
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

        # - update gradients value by that sample

        gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

        

        eta = learningSchedule(epoch*m + i)

        theta = theta - eta * gradients

        err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

        print("Iteration:", epoch*m + i)

        print("Eta:", eta)

        print(theta)

        print("MSE:", err)

        print("-"*10)
nEpochs = 100

batchSize = 10

t0, t1 = 2, 10000000 # learning schedule hyperparameters



def learningSchedule(t): 

    return t0/(t+t1)

c = len(Xb[0])

for i in range(batchSize):

    theta = np.random.randn(c,1) # random initialization

    print('Theta:', theta)

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)



    for epoch in range(nEpochs): 

        # TODO: Implement

        # - randomly take 10 samples (batchSize)

        # - update gradients value by those samples

        # - change the batchSize value to see the difference



        eta = learningSchedule(epoch)

        theta = theta - eta * gradients

        err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

        print(epoch)

        print("Eta:", eta)

        print(theta)

        print("MSE:", err)