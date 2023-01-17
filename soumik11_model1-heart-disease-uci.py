# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/heart.csv")
data
import matplotlib.pyplot as plt
# X = feature values, all the columns except the last column

X = data.iloc[:, :-1]
# y = target values, last column of the data frame

y = data.iloc[:, -1]
# filter out the applicants that got heart disease

admitted = data.loc[y == 1]
# filter out the applicants that din't get heart disease

not_admitted = data.loc[y == 0]
# plots

plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')

plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')

plt.legend()

plt.show()
#preparing the data for our model

X = np.c_[np.ones((X.shape[0], 1)), X]

y = y[:, np.newaxis]

theta = np.zeros((X.shape[1], 1))
#define some functions that will be used to compute the cost

def sigmoid(x):

    # Activation function used to map any real value between 0 and 1

    return 1 / (1 + np.exp(-x))



def net_input(theta, x):

    # Computes the weighted sum of inputs

    return np.dot(x, theta)



def probability(theta, x):

    # Returns the probability after passing through sigmoid

    return sigmoid(net_input(theta, x))
#define the cost and the gradient function

def cost_function(self, theta, x, y):

    # Computes the cost function for all the training samples

    m = x.shape[0]

    total_cost = -(1 / m) * np.sum(

        y * np.log(probability(theta, x)) + (1 - y) * np.log(

            1 - probability(theta, x)))

    return total_cost



def gradient(self, theta, x, y):

    # Computes the gradient of the cost function at the point theta

    m = x.shape[0]

    return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)
def fit(self, x, y, theta):

    opt_weights = fmin_tnc(func=cost_function, x0=theta,

                  fprime=gradient,args=(x, y.flatten()))

    return opt_weights[0]

parameters = fit(X, y, theta)