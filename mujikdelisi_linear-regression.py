# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as spy

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.sample(5)

train.plot.scatter(x='x', y='y')
# strong positive linear correlation
train.describe()
# y shorter
# max value of x is like an outlier
# but what value to append?
# find regression line
# apply a random value using the distribution of x
# at which index y value is NaN?
train[train.isnull()['y']==True]
# at max value of x, y is NaN
# remove index 213
train_ = train[train.index != 213]
train_.describe()
# mean value of x changed drastically after removing index 213
train_.corr('pearson')
# 0.99534
from scipy.stats import zscore
train_.apply(zscore).plot.scatter(x='x', y='y', figsize=[15,10])
# pearson's r is mean value of products of z scores of the variables
train_.apply(zscore)['x'].mul(train_.apply(zscore)['y']).mean()
# the same
# what is the regression line?
# compute regression line
# b = r(stdy/stdx) - std is always positive, so pearson's r gives the sign of intercept
# a = mean(y) - b*mean(x)
# reg = bx + a
b = 0.99534 * (29.109217 / 28.95456)
a = train_['y'].mean() - b * train_['x'].mean()
print(a)
print(b)
train_['linear_regression'] = pd.Series(b * train_['x'] + a)
train_.plot.scatter(x=['x', 'x'], y=['y', 'linear_regression'], c=['b', 'r'], figsize=[15,10])
# now we have regression line, we can find a nice value for missing y
train['y'].iloc[213] = b * train['x'].iloc[213] + a
train['linear_regression'] = pd.Series(b * train['x'] + a)
train.plot.scatter(x=['x', 'x'], y=['y', 'linear_regression'], c=['b', 'r'], figsize=[15,10])
# now load the test set
test = pd.read_csv('../input/test.csv')
test['linear_regression'] = pd.Series(b * test['x'] + a)
test.plot.scatter(x=['x', 'x'], y=['y', 'linear_regression'], c=['b', 'r'], figsize=[15,10])
# graph looks ok