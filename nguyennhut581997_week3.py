# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist

np.random.seed(2)



means = [[2, 2], [4, 2]]

cov = [[.3, .2], [.2, .3]]

N = 10

X0 = np.random.multivariate_normal(means[0], cov, N).T

X1 = np.random.multivariate_normal(means[1], cov, N).T



X = np.concatenate((X0, X1), axis = 1)

y = np.concatenate((np.ones((1, N)), -1*np.ones((1, N))), axis = 1)

# Xbar 

X = np.concatenate((np.ones((1, 2*N)), X), axis = 0)



N = X.shape[1]

d = X.shape[0]

mix_id = np.random.permutation(N)

print(X,y)

xi = X[:, 0].reshape(d, 1)

yi = y[0, 0]

print(xi,yi, xi*yi)
from mpl_toolkits.mplot3d import Axes3D

import math

fig = plt.figure(figsize=(12, 7))

ax = fig.gca(projection='3d')

plt.title('', fontsize=16)



def h(w, x):    

    return np.sign(np.dot(w.T, x))



def has_converged(X, y, w):    

    return np.array_equal(h(w, X), y) 

def perceptron(X, y, w_init):

    w = [w_init]

    N = X.shape[1]

    d = X.shape[0]

    mis_points = []

    dem = 0

    while dem < 1000:

        # mix data 

        mix_id = np.random.permutation(N)

        for i in range(N):

            xi = X[:, mix_id[i]].reshape(d, 1)

            yi = y[0, mix_id[i]]

            if h(w[-1], xi)[0] != yi: # misclassified point

                mis_points.append(mix_id[i])

                w_new = w[-1] + yi*xi 

                w.append(w_new)      

        if has_converged(X, y, w[-1]):

            break

        dem += 1

    return (w, mis_points)



d = X.shape[0]

w_init = np.random.randn(d, 1)

(w, m) = perceptron(X, y, w_init)

print(w[-1])



ax.scatter(X.T[0:10,1],X.T[0:10,2],y.T[0:10,0],marker="o",color='r')

ax.scatter(X.T[11:20,1],X.T[11:20,2],y.T[11:20,0],marker="o",color='g')



ww = w[-1].T

# w0, w1, w2 = ww[0,0], ww[0,1], ww[0,2]

# x11, x12 = -100, 100

# plt.plot([x11, x12], [-(w1*x11 + w0)/w2, -(w1*x12 + w0)/w2], 'k')

plt.plot(ww[0] * X.T[0], ww[0] * X.T[19], marker="o", color="blue")  

ax.set_xlabel('θ_zero')

ax.set_ylabel('θ_one')

ax.set_zlabel('θ_two')

plt.show()
fig = plt.figure(figsize=(12, 7))

ax = fig.gca(projection='3d')

plt.title('Stochastic Gradient Descent', fontsize=16)

theta = w_init



yy = y.T

XX= X.T

m = XX.shape[0]

eta = 2e-8

err = 1

error = 2

while (abs(error - err) > 0.0005):

    error = err;

    gradients = (1/m) * XX.T.dot(XX.dot(theta) - yy)

    theta = theta - eta * gradients

    err = 1/m * math.sqrt((XX.dot(theta)-yy).T.dot(XX.dot(theta)-yy))



print("MSE:", err) 

print(theta)



ax.scatter(X.T[0:10,1],X.T[0:10,2],y.T[0:10,0],marker="o",color='r')

ax.scatter(X.T[11:20,1],X.T[11:20,2],y.T[11:20,0],marker="o",color='g')



plt.plot(theta * X.T[0], theta * X.T[19], marker="o", color="blue")  

ax.set_xlabel('θ_zero')

ax.set_ylabel('θ_one')

ax.set_zlabel('θ_two')

plt.show()
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 

              2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])

y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])

X = np.concatenate((np.ones((1, X.shape[1])), X), axis = 0)

def sigmoid(s):

    return 1/(1 + np.exp(-s))



def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):

    w = [w_init]    

    it = 0

    N = X.shape[1]

    d = X.shape[0]

    count = 0

    check_w_after = 20

    while count < max_count:

        # mix data 

        mix_id = np.random.permutation(N)

        for i in mix_id:

            xi = X[:, i].reshape(d, 1)

            yi = y[i]

            zi = sigmoid(np.dot(w[-1].T, xi))

            w_new = w[-1] + eta*(yi - zi)*xi

            count += 1

            # stopping criteria

            if count%check_w_after == 0:                

                if np.linalg.norm(w_new - w[-check_w_after]) < tol:

                    return w

            w.append(w_new)

    return w

eta = .05 

d = X.shape[0]

w_init = np.random.randn(d, 1)



w = logistic_sigmoid_regression(X, y, w_init, eta)

print(w[-1])
print(sigmoid(np.dot(w[-1].T, X)))