# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import matplotlib.pyplot as plt

%matplotlib inline



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/FuelConsumption.csv')

df.head()
# summarize the data

df.describe()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head(9)
cdf.hist()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS)

plt.xlabel('Engine Size')

plt.ylabel('Co2 Emissions')
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS)

plt.xlabel('Cylinders')

plt.ylabel('Co2 Emissions')
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS)

plt.xlabel('Fuel Consumption')

plt.ylabel('Co2 Emissions')
dataset=np.random.rand (len(df))<0.8

train=cdf[dataset]

test=cdf[~dataset]
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS)

plt.xlabel('Engine Size')

plt.ylabel('CO2 Emissions')
from sklearn import linear_model

from sklearn.linear_model import LinearRegression



reg = LinearRegression()

train_x = np.asanyarray(train[['ENGINESIZE']])

train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit(train_x,train_y)

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("Engine size")

plt.ylabel("Emission")