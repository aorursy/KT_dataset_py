# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# The first thing I want to do is great a basic data set to work from. It will be a simple 2-D data set, and we'll make it a classification 
# dataset of orange/blue points. They'll be Normal Distributions with different means, such that there is some overlap in the middle of the data 
X1_blue = np.random.normal(5.0, 2.0, 100)
X2_blue = np.random.normal(1.0, 2.0, 100)
X1_orange = np.random.normal(1.0, 2.0, 100)
X2_orange = np.random.normal(5.0, 2.0, 100)

# plot for reference
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(X1_blue, X2_blue, s=10, c='blue', marker="o", label='Blue')
ax1.scatter(X1_orange, X2_orange, s=10, c='orange', marker="o", label='Orange')
plt.legend(loc='upper left')
plt.xlabel('X1 input')
plt.ylabel('X2 input')
plt.show()
# Next we'll apply a linear model classification where blue will be coded as 0 and orange as 1. 
# We'll say as a rule that a Y_hat (predicted Y value) less than or equal to 0.5 is Blue and a Y_hat greater than 0.5 is orange
# the fit coefficients will be called beta
Y_blue = np.zeros_like(X1_blue)
Y_orange = np.ones_like(X1_orange)
X1 = np.concatenate([X1_blue, X1_orange])
X2 = np.concatenate([X2_blue, X2_orange])
Y = np.concatenate([Y_blue, Y_orange])

# we assume vectors to be column vectors and each row of matrix X a transposed vector x.T. This means X1 is column 1 of X and X2 is column 2 of X. Y is a column vector
X = np.column_stack((X1,X2))
# so replotting just to make sure we get the same results
# plot for reference
fig = plt.figure()
ax2 = fig.add_subplot(111)
ax2.scatter(X[:99,0], X[:99,1], s=10, c='blue', marker="o", label='Blue=0')
ax2.scatter(X[100:,0], X[100:,1], s=10, c='orange', marker="o", label='Orange=1')
plt.legend(loc='upper left')
plt.xlabel('X[:,0] input')
plt.ylabel('X[:,1] input')
plt.show()

# To apply the linear model, we choose the least squares approach. Beta = (X.transpose * X).inverse * X.transpose * Y