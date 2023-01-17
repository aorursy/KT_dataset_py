import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
#importing library

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import pylab as pl

%matplotlib inline
# Reading data

df = pd.read_csv("/kaggle/input/FuelConsumption.csv")



# take a look at the dataset

df.head()
# summarize the data

df.describe()
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head(9)
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
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")

plt.xlabel("cylinder")

plt.ylabel("co2emmision")

plt.show()
msk = np.random.rand(len(df)) < 0.8

#print(msk)

train = cdf[msk]

test = cdf[~msk]
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')

plt.xlabel("Engine size")

plt.ylabel("Emission")

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

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )