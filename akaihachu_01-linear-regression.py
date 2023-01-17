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



X = data[:,[0, 1]]

y=data[:,[2]]

a = X[:, 0]
m = len(a)

A = 1/(2*m) * (a.T.dot(a))

B = -1/(2*m) * (a.T.dot(y))

C = 1/(2*m) * (y.T.dot(y))

print(A)

print(B)

print(C)



# 2310419.212765957 -382104564.09574467 65591548106.45744
plt.figure(figsize=(10, 5))

theta = np.linspace(-230, 400, 100)

J = A*theta**2 + B*theta + C



# Gives a new shape to an array without changing its data.

plt.plot(theta, J.reshape(-1, 1), color='r')



plt.xlabel("Theta")

plt.ylabel("Cost function")

plt.grid(True)

plt.show()
# TEST CODE

# aaa = np.arange(12).reshape((6, 2))

# print(aaa)

# print(np.c_[np.array([[1,2,3]]), 0, 0, np.array([[4,5,6]])])
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

theta_best = np.array([[89597.9], [139.21067402], [-8738.01911233]])

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

tmp1 = Xb.dot(theta) - y

gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

theta = theta - eta * gradients

err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

print("MSE:", err) 

print(theta)
# TODO: Add early stopping criteria if err doesn't decrease more than epsilon value

# TODO: Check the number of iteration to converge

i = 1

epsilon1=0.0005

e0=err+1+epsilon1

   

while (e0-err>epsilon1):

    e0=err

    tmp1 = Xb.dot(theta) - y

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    i = i + 1

print(i)

print("err:   ",err)
plt.figure(figsize=(10, 5))

plt.title("Batch gradient descent", fontsize=16)



# plt.axis([140, 190, 45, 75])

plt.xlabel("Living area")

plt.ylabel("Price")

xLine = np.array([700, 4500])

theta = np.array([[90000], [120], [-8000]])

yBatchGradient = theta[0,0] + theta[1,0]*xLine

plt.plot(xLine, yBatchGradient, marker="^", color="blue")



eta = 2e-7

gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

theta = theta - eta * gradients

err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))



# TODO: Add early stopping criteria if err doesn't decrease more than epsilon value

# TODO: Check the number of iteration to converge

i = 1

epsilon2=0.0006

e0=err+1+epsilon2

aErr=np.array([])

while (e0-err>epsilon2):

    e0=err

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    aErr=np.append(aErr,err)

    i = i + 1

    yBatchGradient = theta[0,0] + theta[1,0]*xLine

    plt.plot(xLine, yBatchGradient, marker=".", color="r")

    

plt.plot(xLine, yBatchGradient, marker="^", color="black")

plt.plot(a, y, "bo")



plt.grid(True)

plt.show()
plt.figure(figsize=(10, 5))

plt.title("Batch gradient descent", fontsize=16)

plt.xlabel("i")

plt.ylabel("error")

plt.plot(np.array(range(aErr.shape[0])), aErr, "bo")

plt.grid(True)

plt.show()
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(111, projection='3d')

plt.title("Batch gradient descent", fontsize=16)



theta = np.array([[90000], [120], [-8000]])

ax.scatter(theta[0],theta[1],theta[2],marker="^",color='blue') 

eta = 2e-7

gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

theta = theta - eta * gradients

err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

print("MSE:", err)

print(theta)



i = 1

epsilon2=0.0006

e0=err+1+epsilon2

while (e0-err>epsilon2):

    e0=err

    gradients = (2/m) * Xb.T.dot(Xb.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((Xb.dot(theta)-y).T.dot(Xb.dot(theta)-y))

    i = i + 1

    ax.scatter(theta[0],theta[1],theta[2],marker=".",color='r')

ax.scatter(theta[0],theta[1],theta[2],marker="^",color='black') 

ax.set_xlabel('θ_zero')

ax.set_ylabel('θ_one')

ax.set_zlabel('θ_two')

plt.show()
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(X)

scaleX = scaler.transform(X)

scaleXb = np.c_[np.ones((m, 1)), scaleX]

c = len(scaleXb[0])

theta = np.random.randn(c,1)

print(theta)

eta = 0.05
for i in range(1000):

    tmp1 = scaleXb.dot(theta) - y

    gradients = (2/m) * scaleXb.T.dot(scaleXb.dot(theta) - y)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((scaleXb.dot(theta)-y).T.dot(scaleXb.dot(theta)-y))

    if (i>995):

        print(i)

        print(theta)

        print("MSE:", err)

        print("-"*10)
# TODO Question: How to convert theta from scaled value back to normal value? Derive the equation.



max_=np.array([1,np.amax(data[:,0]),np.amax(data[:,1])])

min_=np.array([1,np.amin(data[:,0]),np.amin(data[:,1])])

delta=max_-min_

print(delta)

revert_theta=np.zeros((3,1))

revert_theta[1,0]=theta[1,0]/delta[1]

revert_theta[2,0]=theta[2,0]/delta[2]

revert_theta[0,0]=theta[0,0]-revert_theta[1,0]*min_[1]-revert_theta[2,0]*min_[2]



print(theta)

print("-"*20)

print(revert_theta)



plt.figure(figsize=(10, 5))

plt.title("MinMaxScaler", fontsize=16)



plt.plot(a, y, "bo")

plt.xlabel("Living area")

plt.ylabel("Price")

xLine = np.array([700, 4500])

yLine = 50000 + 60*xLine



#my new

yRevert = revert_theta[0,0] + revert_theta[1,0]*xLine

plt.plot(xLine, yRevert, marker="^", color="r")

plt.grid(True)

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
print(theta)
plt.figure(figsize=(10, 5))

# plt.axis([140, 190, 45, 75])

plt.xlabel("Living area")

plt.ylabel("Price")

xLine = np.array([700, 4500])

plt.title('Stochastic Gradient Descent', fontsize=16)



nEpochs = 20

t0, t1 = 2, 10000000 # learning schedule hyperparameters

def learningSchedule(t): 

    return t0/(t+t1)



c = len(Xb[0])

theta = np.random.randn(c,1)*20 # random initialization

print(theta)



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

        Stoch2D= theta[0] + theta[1]*xLine

        plt.plot(xLine, theta[0] + theta[1]*xLine, marker=".", color="cyan",alpha=(nEpochs-epoch)/nEpochs)

        if (i==0 and epoch==0):

            plt.plot(xLine, theta[0] + theta[1]*xLine, marker=".", color="blue",alpha=(nEpochs-epoch)/nEpochs)

plt.plot(xLine, theta[0] + theta[1]*xLine, marker="^", color="black")    

plt.plot(a, y, "bo")

plt.grid(True)

plt.show()



# colors = [cm(2.*i/15) for i in range(20)]

# colored = [colors[k] for k in df['dest_cluster']]



# ax.scatter(df.dropoff_longitude,df.dropoff_latitude,color=colored,s=0.0002,alpha=1)

fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(111, projection='3d')

plt.title('Stochastic Gradient Descent', fontsize=16)

ax1.scatter(Xb[:,1],Xb[:,2],y.reshape(m),marker="o",color='r')

ax1.scatter(Xb[:,1],Xb[:,2],Xb.dot(theta),marker="*",color='black')



#my new

ax1.set_xlabel('X1_area')

ax1.set_ylabel('X2_room_number')

ax1.set_zlabel('Y_price')

plt.show()
fig = plt.figure(figsize=(12, 7))

ax1 = fig.add_subplot(111, projection='3d')

plt.title('Mini-batch Gradient Descent', fontsize=16)





nEpochs = 100

batchSize = 10

t0, t1 = 2, 10000000 # learning schedule hyperparameters



def learningSchedule(t):

    return t0/(t+t1)

c = len(Xb[0])

theta = np.random.randn(c,1) # random initialization

print("theta random:",theta)

ax1.scatter(theta[0],theta[1],theta[2],marker="*",color='blue')



for epoch in range(nEpochs):

    values=np.array([])

    while m-values.shape[0]>=batchSize:

        Xbatch=np.zeros((0,3))

        ybatch=np.zeros((0,1))

        for i in range(batchSize):

            k=np.random.randint(m)

            while k in values:

                k=np.random.randint(m)

            values=np.append(values,[k])

            Xbatch=np.append(Xbatch,Xb[k].reshape(1,3),axis=0)

            ybatch=np.append(ybatch,y[k].reshape(1,1),axis=0)

        gradients=1/batchSize*Xbatch.T.dot(Xbatch.dot(theta)-ybatch)

        eta = learningSchedule(epoch*batchSize)

        theta = theta - eta * gradients

        err = 1/batchSize * math.sqrt((Xbatch.dot(theta)-ybatch).T.dot(Xbatch.dot(theta)-ybatch))

        ax1.scatter(theta[0],theta[1],theta[2],marker=".",color='r',alpha=(nEpochs-epoch)/nEpochs)

ax1.scatter(theta[0],theta[1],theta[2],marker="o",color='black')        

print("theta final:",theta)

ax1.set_xlabel('θθθθθ_khong')

ax1.set_ylabel('θθθθθ_mot')

ax1.set_zlabel('θθθθθ_hai')

plt.show()
plt.figure(figsize=(10, 5))

# plt.axis([140, 190, 45, 75])

plt.xlabel("Living area")

plt.ylabel("Price")

plt.title('Mini-batch Gradient Descent', fontsize=16)



xLine = np.array([700, 4500])





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

        Xbatch=np.zeros((0,3))

        ybatch=np.zeros((0,1))

        for i in range(batchSize):

            k=np.random.randint(m)

            while k in values:

                k=np.random.randint(m)

            values=np.append(values,[k])

            Xbatch=np.append(Xbatch,Xb[k].reshape(1,3),axis=0)

            ybatch=np.append(ybatch,y[k].reshape(1,1),axis=0)

        gradients=1/batchSize*Xbatch.T.dot(Xbatch.dot(theta)-ybatch)

        eta = learningSchedule(epoch*batchSize)

        theta = theta - eta * gradients

        err = 1/batchSize * math.sqrt((Xbatch.dot(theta)-ybatch).T.dot(Xbatch.dot(theta)-ybatch))

        ax1.scatter(theta[0],theta[1],theta[2],marker=".",color='r')        

        Stoch= theta[0] + theta[1]*xLine

        plt.plot(xLine, theta[0] + theta[1]*xLine, marker=".", color="r",alpha=epoch*0.05)

plt.plot(xLine, theta[0] + theta[1]*xLine, marker="o", color="black")    

plt.plot(a, y, "bo")

print("theta final:",theta)

plt.grid(True)

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
