# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import pylab as pl
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/FuelConsumptionCo2.csv')
data.head(10)
# summary of the data 
data.describe()
cdf = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
cdf.head(10)
df = cdf 
df.hist()
plt.show()
plt.scatter(cdf['ENGINESIZE'], cdf['CO2EMISSIONS'])
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emission')
plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='orange')
plt.xlabel('FUELCONSUMPTION')
plt.ylabel('CO2 Emission')
plt.show()
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')
plt.xlabel('Cylinders')
plt.ylabel('CO2 Emissions')
plt.show()
tmp = np.random.rand(len(data)) < 0.7 
train = cdf[tmp]
test = cdf[~tmp]
print(train.head())
print(test.head())
# Training set data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emission')
plt.show()
from sklearn import linear_model
regr = linear_model.LinearRegression()  # linear  Regression object 
train_x = np.asanyarray(train[['ENGINESIZE']]) # converts input into nd array 
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_p = regr.predict(test_x)
# R2 Score:
print('R2 Score: ', r2_score(test_y_p, test_y))
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_p - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_p - test_y) ** 2))
