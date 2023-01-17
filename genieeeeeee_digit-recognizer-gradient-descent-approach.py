# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Read Training data into pandas dataframe and quick peek

train = pd.read_csv("../input/train.csv", header = 0)

test = pd.read_csv("../input/test.csv", header = 0)

print(str(len(train.columns)) + ' Columns in training set')

print(str(len(train)) + ' Records in training set\n')

print(str(len(test.columns)) + ' Columns in test set')

print(str(len(test)) + ' Records in test set')

print(train.head())

# Training set: 5 x 785 / 784 pixels / first column is number 42000
# Cost function

def computeCost(X, y, theta):

    J = (1/2)*(1/m) * np.dot(np.transpose(X*theta-y), (X*theta-y))

    return J;



# Gradient Descent

def gradientDescent(X, y, theta, alpha, num_iters):

    J_history = np.zeros((num_iters, 1))

    for iter in range(1,num_iters):

        theta = theta - (alpha/m) * np.dot(np.transpose(X), (X*theta - y))

        J_history = computeCost(X,y,theta)

    return J_history;



print('Functions Saved')
# Initialize Gradient Descent

m = len(test)

X = np.hstack((np.ones((m,1)), test.as_matrix(columns=None)))

y = train.as_matrix(columns=None)

theta = np.zeros((m,1))

iterations = 10

alpha = .01

computeCost(X,y,theta)

gradientDescent(X,y,theta,alpha,iterations)