# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import numpy as np

import matplotlib.pyplot as plt

import random

%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Function for generating data

def gen_data(rows, bias, variance):

    X = np.zeros(shape = (rows, 2))

    y = np.zeros(shape = rows)

#     X[i][1] = np.linspace(0, 1, 1)

    for i in range(0, rows):

        X[i][0] = 1

        X[i][1] = random.uniform(0, 1) 

#         X[i][2] = i+1 # use for multi variable if u want n then change X shape or column to n and also 

        #and add values to those column here,...........PEACE OUT

#         y[i] = 5 + X[i][1] 

    

    return X



# Gradient descent main Function

def grad_desc(X, y, alpha, num_iter, theta, m):

    

    X_trans = X.T

    

    for i in range(0, num_iter):

        prediction = np.dot(X, theta)

        error = prediction - y

        cost = np.sum(error**2) / (2*m) # Cost Function though i wont be printing it

#         print("Iteration : %d | Cost : %f" %(i, cost))

        # Calculation for gradient descent starts now

        gradient = np.dot(X_trans, error) / m

        theta = theta - alpha * gradient

    

    return theta, prediction



X = gen_data(100, 25, 10)

X[:,1] = np.linspace(0, 1, 100)

noise = np.random.randn(100)

y = 5*X[:,1] + noise

m, n = np.shape(X)

theta = np.ones(n)

num_iter = 1000 # number of iteration u want to perform feel free to change it as much as u want

alpha = 0.09 #Alpha or so called learning factor

theta, h = grad_desc(X, y, alpha, num_iter, theta, m)

print("\nValues of THETA after "+str(num_iter)+" Iteration is = \n")

print(theta)



plt.scatter(X[:,1], y)

# print(X)

plt.plot(X[:,1], h)