#!/usr/bin/env python3

# -*- coding: utf-8 -*-



import numpy as np

import pandas as pd

import scipy.optimize as op

import matplotlib.pyplot as plt



def sigmoid(z):

    return 1 / (1 + np.exp(-z))



def computeSigmoidCost(theta, X, y):

    m = y.size

    h = sigmoid(X.dot(theta))

    cost = -y.transpose().dot(np.log(h)) - (1-y.transpose()).dot(np.log(1-h))

    return 1/m * cost;



def sigmoidGradient(theta, X, y):

    m = y.size

    h = sigmoid(X.dot(theta))

    grad = 1/m * (h-y).transpose().dot(X)

    return grad



# step 1 - read data

####################

data = pd.read_csv('../input/ex2data1.txt', header=None)

X = data.iloc[:, 0:2].values

y = data.iloc[:, 2].values

m, n = X.shape

y = y.reshape((m,1)); #convert to matrix





# step 2 - plot the data the understand the problem

###################################################

positives = np.where(y[:,0] == 1);

negatives = np.where(y[:,0] == 0);

plt.scatter(X[positives,0], X[positives,1], marker='+', label='Admitted')

plt.scatter(X[negatives,0], X[negatives,1], marker='o', label='Not admitted')

plt.xlabel('Exam 1 score')

plt.ylabel('Exam 2 score')

plt.legend()

plt.show()





# step 3 - compute cost

#######################



# add the intercept term

X = np.concatenate((np.ones((m,1), dtype=np.int), X.reshape(m, n)), axis=1)



# fitting parameters with theta zeros

initial_theta = np.zeros((n+1,1))

cost = computeSigmoidCost(initial_theta, X, y)

grad = sigmoidGradient(initial_theta, X, y)

print('Cost at initial theta (zeros): {}'.format(cost))

print('Expected cost (approx): 0.693')

print('Gradient at initial theta (zeros): {}'.format(grad))

print('Expected gradients (approx): -0.1000 -12.0092 -11.2628')



# non-zero theta

test_theta = np.array([[-24], [0.2], [0.2]])

cost = computeSigmoidCost(test_theta, X, y)

grad = sigmoidGradient(test_theta, X, y)

print('Cost at test theta: {}'.format(cost))

print('Expected cost (approx): 0.218')

print('Gradient at test theta: {}'.format(grad))

print('Expected gradients (approx): 0.043 2.566 2.647')



# step 4 - optimizing theta

###########################

options = {'full_output': True, 'maxiter': 400}



theta, cost, _, _, _ = op.fmin(lambda t: computeSigmoidCost(t, X, y), initial_theta, **options)



# print the decision boundary

positives = np.where(y[:,0] == 1);

negatives = np.where(y[:,0] == 0);

plt.scatter(X[positives,1], X[positives,2], marker='+', label='Admitted')

plt.scatter(X[negatives,1], X[negatives,2], marker='o', label='Not admitted')

px = np.array([np.min(X[:,1])-2, np.max(X[:,2])+2])

py = (-1 / theta[2]) * (theta[1]*px + theta[0])

plt.plot(px, py)

plt.xlabel('Exam 1 score')

plt.ylabel('Exam 2 score')

plt.legend()

plt.show()





# step 5 - predicting

#####################

prob = sigmoid(np.array([1, 45, 85]).dot(theta));

print('For a student with scores 45 and 85, we predict an admission probability of {}'.format(prob))

print('Expected value (approx): 0.775')




