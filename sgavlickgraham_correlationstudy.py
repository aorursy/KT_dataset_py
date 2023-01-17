# import necessary tools

from numpy import mean

from numpy import std

from numpy.random import randn

from numpy.random import seed

from numpy import cov

from scipy.stats import pearsonr

from scipy.stats import spearmanr

import matplotlib

from matplotlib import pyplot

# seed random number generator

seed(1)
# generate related variables



# prepare data

# adding 100 makes the mean 100 and multiplying by 20 makes the std 20.

data1 = 20 * randn(1000) + 100 

# add "noise" to the first data. The noise has a mean of 50 and std of 10.

data2 = data1 + (10 * randn(1000) + 50)

# summarize

print('data1: mean=%.3f stdv=%.3f' % (mean(data1), std(data1)))

print('data2: mean=%.3f stdv=%.3f' % (mean(data2), std(data2)))
# plot

matplotlib.rcParams['figure.figsize'] = [12.0, 8.0]

pyplot.scatter(data1, data2)

pyplot.show()
# calculate covariance matrix

covariance = cov(data1, data2)

print(covariance)
# calculate the Pearson's correlation between two variables

corr, _ = pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)
# calculate the spearmans's correlation between two variables

corr, _ = spearmanr(data1, data2)

print('Spearmans correlation: %.3f' % corr)