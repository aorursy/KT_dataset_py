import numpy as np

import pandas as pd

import os
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
# Load Iris data

data = load_iris()



x = data['data']

y = data['target']
print(x)

print(y)
print(x.shape)
# Since PCA is an unsupervised method, we will not be using the target variable y

# Scale the data such that mean = 0 and standard deviation = 1

from sklearn.preprocessing import scale

x_s = scale(x, with_mean=True, with_std=True, axis=0)

print(x_s)
# calculate correlation matrix

x_c = np.corrcoef(x_s.T)

print(x_c)
# Find eigen value and eigen vector from correlation matrix

import scipy

eig_val, r_eig_vec = scipy.linalg.eig(x_c)

print('Eigen values = ', eig_val)

print('Eigen vectors = ', r_eig_vec)

# Select the first two eigen vectors

w = r_eig_vec[:, 0:2]

print('First two eigen vectors = ', w)
# # Project the dataset from 4 Dimension to 2 Dimension 

# # using the right eigen vector

x_rd = x_s.dot(w)

# print(x_rd)



# Scatter plot the new two dimensions

plt.figure(1)

plt.scatter(x_rd[:, 0], x_rd[:,1], c=y)

plt.xlabel("Component 1")

plt.ylabel("Component 2")
# Ways to help us select how many components should we include.

# In our recipe, we included only two. The following are a

# list of ways to select the components more empirically:



# The Eigenvalue criterion:

# An Eigenvalue of one would mean that the component would explain about one

# variable's worth of variability. So, according to this criterion, a component should

# at least explain one variable's worth of variability. We can say that we will include

# only those Eigenvalues whose value is greater than or equal to one. Based on our

# data set we can set the threshold. In a very large dimensional dataset including

# components capable of explaining only one variable may not be very useful.

print("Component, Eigen Value, % of Variance, Cummulative % ")

cum_per = 0

per_var = 0

for i, e_val in enumerate(eig_val):

    per_var = round((e_val / len(eig_val)), 3)

    cum_per += per_var

    print(('%d, %0.2f, %0.2f, %0.2f') % (i +1, e_val, per_var * 100, cum_per* 100))