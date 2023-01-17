import pandas as pd
import numpy as np
data = pd.read_csv('/kaggle/input/ex1data1.txt', sep = ',', header = None)
data.info()
data.head()
data.describe()
data = data.rename(columns = {0:"Population", 1:"Profit"})
data.head()
data.plot.scatter(x = 'Population', y = 'Profit')
npData = data.to_numpy()
X = npData[:,0]
y = npData[:,1]
X = X.reshape((X.shape[0],1))
y = y.reshape((y.shape[0],1))
theta0 = np.ones((X.shape[0],1))
X = np.concatenate((theta0, X), axis = 1)
theta = np.zeros((2,1))
iterations = 1500
alpha =0.01
m = X.shape[0]

def calculateJTheta(X,y):
    JTheta = 1/ (2 * m ) * np.sum(np.square(np.dot(X, theta) - y))
    # print (np.dot(X, theta) - y).shape)
    return JTheta / (2 * m)
# np.sum(np.multiply(np.dot(X,theta) - y, X), axis = 0)
# np.dot(X,theta)
# X.shape
# theta.shape
# np.multiply(np.dot(X,theta) - y, X)
# temp = np.array([[]])
# temp = np.concatenate((temp,np.array([1,2])), axis = 0)
# temp = np.concatenate((temp,np.array([3,4])), axis = 0)
# len(theta)
JThetaArr = np.zeros((iterations,len(theta)))

def deltaJ(X, y):
    JTheta = 1/ (2 * m ) * np.sum(np.square(np.dot(X, theta) - y))
    return 1/ m * np.sum(np.multiply(np.dot(X,theta) - y, X), axis = 0)

for i in range(iterations):
    JThetaArr[i] = 1/ (2 * m ) * np.sum(np.square(np.dot(X, theta) - y))
    theta = theta - alpha * deltaJ(X,y).reshape((2,1))
yPred = np.dot(X, theta)

import matplotlib.pyplot as plt

plt.scatter(npData[:,0], y )
plt.plot(npData[:,0], yPred, color = 'red')
plt.show()
# cs = plt.contour(JThetaArr, levels = [10,30,50])
# cs.cmap.set_over('red')
# cs.cmap.set_under('blue')
# cs.changed()