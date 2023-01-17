# Importing necessary classes

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import sklearn.cross_validation as cv

import sklearn.linear_model as lm

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score



# Import Input File

train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')



# Delete Observations with Missing Values

train=train.dropna()

test=test.dropna()



# Separate Independent and Dependent Variables

x_train=train.iloc[:,:-1].values

y_train=train.iloc[:,-1].values

x_test=test.iloc[:,:-1].values

y_test=test.iloc[:,-1].values



# Create Linear Regression Object and fit it to Train Dataset

regressor = LinearRegression()

regressor.fit(x_train, y_train)



# Apply the model on Test dataset and Predict the Dependent Variable

y_pred = regressor.predict(x_test)



# The coefficients

print('Coefficients: \n', regressor.coef_)

# The mean squared error

print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))

# Explained variance score: 1 is perfect prediction

print('R-Squared: %.2f' % r2_score(y_test, y_pred))



# Visualising the Training set results

plt.scatter(x_train, y_train, color = 'red')

plt.plot(x_train, regressor.predict(x_train), color = 'blue')

plt.title('X vs Y (Training set)')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()



# Visualising the Test set results

plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_train, regressor.predict(x_train), color = 'blue')

plt.title('X vs Y (Test set)')

plt.xlabel('X')

plt.ylabel('Y')

plt.show()
