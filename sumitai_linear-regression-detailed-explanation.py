import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt 

sns.set()

x = [1,2,4,3,5, 6, 7, 8, 9, 10]

y = [1,3,3,2,5, 7, 6, 8, 11, 10]

plt.plot(x,y, '*')

plt.xlabel('X')

plt.ylabel('Y')

plt.title('X vs Y plot')
# calculate the mean of x and y 

import numpy as np 

mean_y = np.mean(y)

mean_x = np.mean(x)

print('mean of y: ', mean_y)

print('mean of x: ', mean_x)
y_meany = y - mean_y 

x_meanx = x - mean_x

print('Residual of each y value from the mean.: ', y_meany)

print('Residual of each x value from the mean.: ', x_meanx)

mean_mul = y_meany*x_meanx

mean_mul
squareX = (x_meanx)**2

sum_X = np.sum(squareX)

sum_X

sum_meanmul = np.sum(mean_mul)

sum_meanmul
B1 = np.sum(mean_mul)/np.sum(squareX)

B1
np.mean(y)
B0 = np.mean(y) - B1*np.mean(x)

print('B0: ', B0)
y1 =  np.array(x)*B1 +B0

y1
plt.plot(x,y1, '-o')

plt.plot(x,y, 'o')
from scipy.stats import pearsonr

corr, _ = pearsonr(x, y)

B1 = corr*(np.std(x)/np.std(y))

B1
B0 = np.mean(y) - B1*np.mean(x)

print('B0: ', B0)
y1 =  np.array(x)*B1 +B0

plt.plot(x,y1, '-o')

plt.plot(x,y, 'o')
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())
rmse(y1,y)