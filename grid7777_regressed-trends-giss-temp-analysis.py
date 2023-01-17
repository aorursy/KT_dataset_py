#Import Packages: Data Manipulation

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

#Import Regression Imports

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

import os

fileName = '../input/annual_csv.csv'

data_annual = pd.read_csv(fileName)

data_annual.head()

data_annual.describe()
#SEPERATING GCAG and GISTEMP data.

data_GCAG = data_annual[data_annual.Source == 'GCAG']

data_GISTEMP = data_annual[data_annual.Source == 'GISTEMP']



#GCAC data seperated yearwise and tempreaturewise

data_GCAC_year = data_GCAG['Year']

data_GCAC_temp = data_GCAG['Mean']



#GISTEMP data seperated yearwise and tempreaturewise

data_GISTEMP_year = data_GISTEMP['Year']

data_GISTEMP_temp = data_GISTEMP['Mean']



#Reduce to values

x = data_GCAC_year.values

y = data_GCAC_temp.values



x0 = data_GCAC_year.values

y0 = data_GCAC_temp.values
X = x.reshape(x.shape[0],-1)

Y = y.reshape(y.shape[0],-1)



X0 = x0.reshape(x.shape[0],-1)

Y0 = y0.reshape(y.shape[0],-1)
#Fit the model

temperature_regression_model = LinearRegression()

temperature_regression_model.fit(X,Y)

Y_prediction = temperature_regression_model.predict(X)
# data points

plt.scatter(X, Y, s=10)

plt.xlabel('year')

plt.ylabel('y')

plt.title('GCAG Data Temperature Trend')



# predicted values

plt.plot(X, Y_prediction, color='r')

plt.show()
rmse =  mean_squared_error(Y,Y_prediction)

r2 = r2_score(Y, Y_prediction)



print('Slope:' ,temperature_regression_model.coef_)

print('Intercept:', temperature_regression_model.intercept_)

print('Root mean squared error: ', rmse)

print('R2 score: ', r2)
temperature_regression_model.fit(X0,Y0)

Y0_prediction = temperature_regression_model.predict(X0)
# data points

plt.scatter(X0, Y0, s=10, color = 'g')

plt.xlabel('year')

plt.ylabel('y')

plt.title('GISTMP Data Temperature Trend')



# predicted values

plt.plot(X0, Y0_prediction, color='r')

plt.show()
print('Slope:' ,temperature_regression_model.coef_)

print('Intercept:', temperature_regression_model.intercept_)

print('Root mean squared error: ', rmse)

print('R2 score: ', r2)