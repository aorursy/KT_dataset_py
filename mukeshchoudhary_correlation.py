from numpy.random import seed

import numpy as np

from numpy.random import randn

import matplotlib.pyplot as plt

seed(1)

data1= 20*randn(1000)+ 100 #std*(Gaussian values with a mean of 0 and a standard deviation of 1)+ mean

data2= data1+ (10*randn(1000)+50)# noise added

print('data1: mean:%.3f stdv:%.3f'%(np.mean(data1),np.std(data1)))

print('data2: mean:%.3f stdv:%.3f'%(np.mean(data2),np.std(data2)))

plt.scatter(data1, data2)

plt.show()
covariance= np.cov(data1, data2)

print(covariance)
from scipy.stats import pearsonr

corr, _= pearsonr(data1, data2)

print('Pearsons correlation: %.3f' % corr)
from scipy.stats import spearmanr

corr, _= spearmanr(data1, data2)

print('Spearman corr:%.3f'%corr)