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

data = pd.read_csv(ftPath, sep = ',', header = None)

data.columns = ['Living Area', 'Bedrooms', 'Price']



X0 = data['Living Area'].values.reshape(-1, 1)

X1 = data['Bedrooms'].values.reshape(-1, 1)

X = np.concatenate((X0,X1), axis = 1)

# X = data['Living Area'].values.reshape(-1, 1)

Y = data['Price'].values.reshape(-1, 1)

# print(X)

# print(Y)

print(len(X), len(Y))

from sklearn.linear_model import LinearRegression 

from mpl_toolkits.mplot3d import Axes3D

lin_reg = LinearRegression()

lin_reg.fit(X, Y)

Y_pred = lin_reg.predict(X)



print(lin_reg.intercept_, lin_reg.coef_)



X0 = data['Living Area']

X1 = data['Bedrooms']

Y = data['Price']

_3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

_3d_figure.plot(X0, X1, Y)

# plt.scatter(X, Y)



# Draw the linear regression model

_3d_figure.plot(X0, X1, Y_pred, color = 'red')



plt.show()
m = len(X)

Xb = np.c_[np.ones((m, 1)), X]

y = np.c_[Y]

c = len(Xb[0])

theta = np.random.randn(c,1)

eta = 2e-7

err = 1

error = 2

while (abs(error - err) > 0.0005):

    error = err;

    tmp1 = Xb.dot(theta) - y

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

print("MSE:", err) 

print(theta)



Y_pred = Xb.dot(theta)

Y_pred = Y_pred.reshape(1,-1)

Y = data['Price']

_3d_figure = plt.figure(figsize = (15, 10)).gca(projection = '3d')

_3d_figure.plot(X0, X1, Y)



# Draw the linear regression model

_3d_figure.plot(X0, X1, Y_pred[0], color = 'red')



plt.show()
fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(111, projection='3d')

plt.title('Stochastic Gradient Descent', fontsize=16)



nEpochs = 20

t0, t1 = 2, 10000000 # learning schedule hyperparameters

def learningSchedule(t): 

    return t0/(t+t1)



c = len(Xb[0])

theta = np.random.randn(c,1)*20 # random initialization

print(theta)

ax.scatter(theta[0],theta[1],theta[2],marker="o",color='blue') 

for epoch in range(nEpochs):

    values=np.array([])

    for i in range(m):

        k=np.random.randint(m)

        while k in values:

            k=np.random.randint(m)

        values=np.append(values,[k])

        gradients=1/m*(Xb[k].dot(theta) - y[k])*Xb[k].reshape(3,1)

        eta = learningSchedule(epoch*m + i)

        theta = theta - eta *gradients 

        err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

        ax.scatter(theta[0],theta[1],theta[2],marker=".",color='r')

ax.scatter(theta[0],theta[1],theta[2],marker="o",color='black') 

ax.set_xlabel('θ_zero')

ax.set_ylabel('θ_one')

ax.set_zlabel('θ_two')

plt.show()
fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(111, projection='3d')

plt.title('Mini-batch Gradient Descent', fontsize=16)

ax1.scatter(Xb[:,1],Xb[:,2],y.reshape(m),marker="o",color='r')

ax1.scatter(Xb[:,1],Xb[:,2],Xb.dot(theta),marker="^",color='blue')



#my new

ax1.set_xlabel('X1_area')

ax1.set_ylabel('X2_room_number')

ax1.set_zlabel('Y_price')

plt.show()
plt.figure(figsize=(10, 5))

# plt.axis([140, 190, 45, 75])

plt.xlabel("Living area")

plt.ylabel("Price")

plt.title('Mini-batch Gradient Descent', fontsize=16)

a = X[:, 0]



xLine = np.array([700, 4500])



print(xLine)

nEpochs = 20

batchSize = 10

t0, t1 = 2, 10000000 # learning schedule hyperparameters



def learningSchedule(t):

    return t0/(t+t1)

c = len(Xb[0])

theta = np.array([[100],[1],[10]]) # random initialization

print("theta random:",theta)

plt.plot(xLine, theta[0] + theta[1]*xLine, marker="o", color="blue")    

for epoch in range(nEpochs):

    values=np.array([])

    while m-values.shape[0]>=batchSize:

        Xbatch=np.array([])

        ybatch=np.array([])

        for i in range(batchSize):

            k=np.random.randint(m)

            while k in values:

                k=np.random.randint(m)

            values=np.append(values,[k])

            Xbatch=np.append(Xbatch,np.array(Xb[k]))

            ybatch=np.append(ybatch,np.array(y[k]))

        Xbatch1=Xbatch.reshape(batchSize,3)

        ybatch1=ybatch.reshape(-1,1)

        gradients=1/batchSize*Xbatch1.T.dot(Xbatch1.dot(theta)-ybatch1)

        eta = learningSchedule(epoch*batchSize)

        theta = theta - eta * gradients

        ax1.scatter(theta[0],theta[1],theta[2],marker=".",color='r')        

        err = 1/batchSize * math.sqrt((Xbatch1.dot(theta)-ybatch1).T.dot(Xbatch1.dot(theta)-ybatch1))



        Stoch= theta[0] + theta[1]*xLine

        plt.plot(xLine, Stoch, marker=".", color="r",alpha=epoch*0.05)

plt.plot(xLine, theta[0] + theta[1]*xLine, marker="o", color="black")    

plt.plot(a, y, "bo")

print("theta final:",theta)

plt.grid(True)

plt.show()