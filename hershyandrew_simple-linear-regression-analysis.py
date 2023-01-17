import numpy as np

import pandas as pd

from scipy import stats

import statsmodels.api as sm 

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression



data = pd.read_csv('../input/Cricket_chirps.csv')

data
X = data['X']

Y = data['Y']

print(data['X'],data['Y'])
plt.scatter(X,Y)

plt.axis([0,95,0,25])

plt.ylabel('Chirps/second')

plt.xlabel('Temperature in F')

plt.show()
X1 = sm.add_constant(X)

X1
reg = sm.OLS(Y,X1).fit()

reg.summary()
slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
slope
intercept
r_value**2
p_value
std_err
def calc_crick(x):

    return intercept + (x * slope)



calc_crick(85)
def fitline(b):

    return intercept + slope * b



line = fitline(X)



plt.scatter(X,Y)

plt.plot(X,line)

plt.show()