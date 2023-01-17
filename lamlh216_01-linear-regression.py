#Bài tập ngày 2 - Linear Regression

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



### Show data

# re-readData

dataVisual = pd.read_csv(ftPath, sep = ',', header = None)

dataVisual.columns = ['Living Area', 'Bedrooms', 'Price']



# Print out first 5 rows to get the imagination of data

dataVisual.head()
from sklearn.linear_model import LinearRegression
X1 = dataVisual['Living Area'].values.reshape(-1, 1)

X2 = dataVisual['Bedrooms'].values.reshape(-1, 1)

X = np.column_stack((X1,X2))

Y = dataVisual['Price'].values.reshape(-1, 1)



# Train the model

model = LinearRegression()

model.fit(X, Y)



# Predict with the same input data

Y_pred = model.predict(X)



print(model.predict([[2000, 2]]))



# Visualize the data

plt.scatter(X1, Y)

plt.plot(X, Y_pred, color = 'red')

plt.show()

plt.scatter(X2, Y)

plt.plot(X, Y_pred, color = 'red')

plt.show()
# X vector: [[living area, Bedrooms]]

X = data[:,[0, 1]]



# y: [[price]]

y = data[:,[2]]



# a: [Living area]

a = X[:, 0]

print(a)

m = len(a)

A = 1/(2*m) * (a.T.dot(a))

B = -1/(m) * (a.T.dot(y))

C = 1/(2*m) * (y.T.dot(y))

print(A)

print(B)

print(C)



# 2310419.212765957 -382104564.09574467 65591548106.45744
def getCostFunc():

    plt.figure(figsize=(10, 5))

    theta = np.linspace(-230, 560, 100)

    J = A*theta**2 + B*theta + C



    # Gives a new shape to an array without changing its data.

    plt.plot(theta, J.reshape(-1, 1), color='r')



    plt.xlabel("Theta")

    plt.ylabel("Cost function")

    plt.grid(True)

    return plt



getCostFunc().show()
# Solve the problem for theta0, theta1

from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()

lin_reg.fit(X, y)

print(lin_reg.intercept_)

print(lin_reg.coef_)
from sklearn.linear_model import LinearRegression 

lin_reg = LinearRegression()

lin_reg.fit(X, y)

print(lin_reg.intercept_, lin_reg.coef_)
m = len(X)

# Translates slice objects to concatenation along the second axis.

# add slice 1..1 to X

Xb = np.c_[np.ones((m, 1)), X]

theta_best = np.array([[89597.9], [139.21067402], [-8738.01911233]]) # normally, this should be a random number

error = 1/m * math.sqrt((Xb.dot(theta_best)-y).T.dot(Xb.dot(theta_best)-y))

print("MSE:", error)

print(theta_best)
c = len(Xb[0])

# Return a sample

# (or samples) from the “standard normal” distribution.



theta = np.random.randn(c,1)

print(c)

print(theta)

eta = 2e-7
# Each iteration

#tmp1 = Xb.dot(theta) - y

#gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

#theta = theta - eta * gradients

#err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

#print("MSE:", err) 

#print(theta)
# TODO: Add early stopping criteria if err doesn't decrease more than epsilon value

#### DONE: stop condition: err0 - err < epsilon

# TODO: Check the number of iteration to converge

#### DONE: i logged at the end of loop

i = 1

epsilon_GD=0.00031

print(theta)

# init error must be biggest so we can keep it loop at least 1 time:

err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

err0 = err + epsilon_GD + 10 # 10 or any other number is OK, wee just keep it bugger so it can loop at least 1 time



def plotDataIntoCostFunc(theta):

    print(theta)

    thetaA = theta[1][0]

    J0 = A*thetaA**2 + B*thetaA + C



    newCostFunc = getCostFunc()

    newCostFunc.scatter(thetaA, J0)

    newCostFunc.show()

   

while (True):

    e0=err

    tmp1 = Xb.dot(theta) - y

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    plotDataIntoCostFunc(theta)

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    i = i + 1

    if err0 - err < epsilon_GD:

        break

    else:

        print(i,'- err decreasement: ',(err0 - err))

        err0 = err

        



#print result

thetaResult=theta[:, 0]

print("loop:  ",i)

print("err:   ",err)

print("thetaResult: ",thetaResult)

# predict function h(x) = y

def H_func(features = [['Living area', 'bedrooms']]):

    featuresBar=features[0]

    featuresBar.insert(0, 1) # add number 1 into features vector

    print("featuresBar", featuresBar)

    return thetaResult[0] + thetaResult[1]*featuresBar[1] + thetaResult[2]*featuresBar[2]



print("predict result: ", H_func([[2000, 2]]))