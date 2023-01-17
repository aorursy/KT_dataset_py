import matplotlib.pyplot as plt

import pandas as pd

import pylab as pl

import numpy as np

%matplotlib inline
df = pd.read_csv('../input/fuelconsumptionco2/FuelConsumptionCo2.csv')



# take a look at the dataset

df.head()
# Checking the shape of the data frame

df.shape
# Checking the infomration

df.info()
#Checking columns

df.columns
# descriptive statistics

df.describe().T
# Checking null count

import missingno as msno

p=msno.bar(df)

# The CO2 Emission is what is need to be calculated, so it is important to plot each column vs Emission. 

# With that it is easy to understand the data when it is visualized.

fig, ax = plt.subplots(2, 2, figsize=(20, 10), sharey = True)



ax[0,0].set_ylabel("CO2 Emission")

ax[0,0].scatter(df["FUELCONSUMPTION_CITY"], df["CO2EMISSIONS"])

ax[0,0].set_xlabel("FUELCONSUMPTION_CITY")



ax[0,1].scatter(df["FUELCONSUMPTION_HWY"], df["CO2EMISSIONS"])

ax[0,1].set_xlabel("FUELCONSUMPTION_HWY")



ax[1,0].set_ylabel("CO2 Emission")

ax[1,0].scatter(df["ENGINESIZE"], df["CO2EMISSIONS"])

ax[1,0].set_xlabel("ENGINESIZE")



ax[1,1].scatter(df["CYLINDERS"], df["CO2EMISSIONS"])

ax[1,1].set_xlabel("CYLINDERS")



#plt.savefig('images/visualData.png', dpi=400, bbox_inches="tight")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

cdf.head(9)
plt.figure(figsize = (15,10))

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
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='blue')

plt.xlabel("Cylinders")

plt.ylabel("Emission")

plt.show()
msk = np.random.rand(len(df)) < 0.8

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

plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], 'red')

plt.xlabel("Engine size")

plt.ylabel("Emission")
from sklearn.metrics import r2_score



test_x = np.asanyarray(test[['ENGINESIZE']])

test_y = np.asanyarray(test[['CO2EMISSIONS']])

test_y_hat = regr.predict(test_x)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_hat , test_y) )
from sklearn.preprocessing import PolynomialFeatures

from sklearn import linear_model

train_x = np.asanyarray(train[['ENGINESIZE']])

train_y = np.asanyarray(train[['CO2EMISSIONS']])



test_x = np.asanyarray(test[['ENGINESIZE']])

test_y = np.asanyarray(test[['CO2EMISSIONS']])





poly = PolynomialFeatures(degree=2)

train_x_poly = poly.fit_transform(train_x)

train_x_poly
regr = linear_model.LinearRegression()

train_y_ = regr.fit(train_x_poly, train_y)

# The coefficients

print ('Coefficients: ', regr.coef_)

print ('Intercept: ',regr.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')

XX = np.arange(0.0, 10.0, 0.1)

yy = regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2)

plt.plot(XX, yy, 'red' )

plt.xlabel("Engine size")

plt.ylabel("Emission")
from sklearn.metrics import r2_score



test_x_poly = poly.fit_transform(test_x)

test_y_ = regr.predict(test_x_poly)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
poly3 = PolynomialFeatures(degree=3)

train_x_poly3 = poly3.fit_transform(train_x)



regr3 = linear_model.LinearRegression()

regr3.fit(train_x_poly3, train_y)



plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')

yy = regr3.intercept_[0]+ regr3.coef_[0][1]*XX+ regr3.coef_[0][2]*np.power(XX, 2)+ regr3.coef_[0][3]*np.power(XX, 3)

plt.plot(XX, yy, 'red' )

plt.xlabel("Engine size")

plt.ylabel("Emission")



test_x_poly = poly3.fit_transform(test_x)

test_y_ = regr3.predict(test_x_poly)



print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))

print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))

print("R2-score: %.2f" % r2_score(test_y_ , test_y) )
