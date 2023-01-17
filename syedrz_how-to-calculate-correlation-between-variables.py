# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# generate related variables

from numpy import mean

from numpy import std

from numpy.random import randn

from numpy.random import seed

from matplotlib import pyplot

# seed random number generator

seed(1)

# prepare data

data1 = 20 * randn(1000) + 100

data2 = data1 + (10 * randn(1000) + 50)

# summarize

print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))

print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))

# plot

#A scatter plot of the two variables is created. Because we contrived the dataset, 

#we know there is a relationship between the two variables. 

#This is clear when we review the generated scatter plot where we can see an increasing trend.

pyplot.scatter(data1, data2)

pyplot.show()
# calculate the covariance between two variables

from numpy.random import randn

from numpy.random import seed

from numpy import cov

# seed random number generator

seed(1)

# prepare data

data1 = 20 * randn(1000) + 100

data2 = data1 + (10 * randn(1000) + 50)

# calculate covariance matrix

covariance = cov(data1, data2)

print(covariance)
# calculate the Pearson's correlation between two variables

from numpy.random import randn

from numpy.random import seed

from scipy.stats import pearsonr

# seed random number generator

seed(1)

# prepare data

data1 = 20 * randn(1000) + 100

data2 = data1 + (10 * randn(1000) + 50)

# calculate Pearson's correlation

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)
# calculate the spearmans's correlation between two variables

from numpy.random import randn

from numpy.random import seed

from scipy.stats import spearmanr

# seed random number generator

seed(1)

# prepare data

data1 = 20 * randn(1000) + 100

data2 = data1 + (10 * randn(1000) + 50)

# calculate spearman's correlation

corr, _ = spearmanr(data1, data2)

print('Spearmans correlation: %.3f' % corr)