import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import random

random.seed(20)

data = np.array([[i, i**2 * random.uniform(0.8, 1.3) + 3] for i in range(-15, 21) ])
x = data[:,0].reshape((data.shape[0], 1))

y = data[:,1].reshape((data.shape[0], 1))
def plotData(x, y, **kwargs):

    plt.figure(figsize=(12,7))

    plt.yscale("linear")

    plt.scatter(x, y, **kwargs)
plotData(x, y)
# x = (m, n) | y = (m, 1), t = (1, n), where n = features and m = features

def hyp(x, theta):

    return x.dot(theta.T)



def get_X(x):

    m = x.shape[0]

    return np.concatenate((np.ones((m, 1)), x), axis=1)

    

def cost(x, y, theta):

    m = x.shape[0]

    x_0 = np.ones((m, 1))

    h_of_x = hyp(x, theta)

    return 1/(2 * m) * (h_of_x - y).sum() ** 2



def LinReg(x, y, theta, alpha=0.01, n_iters=400):

    m = x.shape[0]

    X = np.concatenate((np.ones((m, 1)), x), axis=1)

    costs = np.zeros(n_iters)

    

    

    for i in range(n_iters):

        h_of_x = hyp(X, theta)

        gradient = alpha/m * (h_of_x - y).T.dot(X)

        

        theta -= gradient

        costs[i] = cost(X, y, theta)

    return theta, costs

t = np.zeros((1, 2))

theta, costs = LinReg(x, y, t, 0.01)

plt.plot(costs)
y_pred = hyp(get_X(x), theta)



plt.figure(figsize=(12,7))

plt.yscale("linear")

plt.scatter(x, y)

plt.plot(x, y_pred, c="r")
# x = (m, n) | y = (m, 1), t = (1, n), where n = features and m = features

def p_hyp(x, theta):

    one = x.dot(theta.T)

    square = np.square(x).dot(theta.T)

    return one + square



def PolReg(x, y, theta, alpha=0.01, n_iters=400):

    m = x.shape[0]

    X = np.concatenate((np.ones((m, 1)), x), axis=1)

    costs = np.zeros(n_iters)

    

    

    for i in range(n_iters):

        h_of_x = p_hyp(X, theta)

        gradient = alpha/m * (h_of_x - y).T.dot(X)

        theta -= gradient

        costs[i] = cost(X, y, theta)

    return theta, costs

t = np.zeros((1, 2))

theta_p, costs = PolReg(x, y, t, 0.002, 2000)

print("Theta", theta)

plt.plot(costs)
y_pred = p_hyp(get_X(x), theta_p)



plt.figure(figsize=(12,7))

plt.yscale("linear")

plt.scatter(x, y)

plt.plot(x, y_pred, c="r")