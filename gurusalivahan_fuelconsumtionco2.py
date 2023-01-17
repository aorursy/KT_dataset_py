# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import pandas as pd

import pylab as pl

import numpy as np

%matplotlib inline
df = pd.read_csv(r"/kaggle/input/fuelconsump/FuelConsumptionCo2.csv")

#  dataset

df.head()



#data summarization

df.describe()
#some features to explore

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head(10)
viz = cdf[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]

viz.hist()

plt.show()
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS,  color='blue')

plt.xlabel("FUELCONSUMPTION_COMB")

plt.ylabel("Emission")

plt.show()
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')

plt.xlabel("Engine size")

plt.ylabel("Emission")

plt.show()
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS,  color='blue')

plt.xlabel("Cylinders")

plt.ylabel("CO2Emission")

plt.show()
# Splitting the dataset into the Training set and Test set



msk = np.random.rand(len(df)) < 0.8

train = cdf[msk]

test = cdf[~msk]
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')

plt.xlabel("Engine size")

plt.ylabel("CO2Emission")

plt.show()
plt.scatter(test.ENGINESIZE, test.CO2EMISSIONS,  color='blue')

plt.xlabel("Engine size")

plt.ylabel("CO2Emission")

plt.show()
from sklearn import linear_model

regr = linear_model.LinearRegression()

train_x = np.asanyarray(train[['ENGINESIZE']])

train_y = np.asanyarray(train[['CO2EMISSIONS']])

regr.fit (train_x, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')

plt.xlabel("Engine size")

plt.ylabel("Emission")
from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['ENGINESIZE']])

test_y = np.asanyarray(test[['CO2EMISSIONS']])

test_y_ = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_ , test_y) )