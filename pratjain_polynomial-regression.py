import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os

#print(os.listdir("../input")).
# Importing the dataset.

data = pd.read_csv('../input/Position_Salaries.csv')

x = data.iloc[:, 1:2].values

y = data.iloc[:, 2].values
# Splitting the dataset into the Training set and Test set

# Because we have so small number of observations we don't want to split the data

#from sklearn.cross_validation import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x,y)
# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly,y)
# Visualising the Linear Regression Results

plt.scatter(x,y, color = 'red')

plt.plot(x, lin_reg.predict(x), color = 'blue')

plt.title('Linear Regression')

plt.xlabel('Position')

plt.ylabel('Salary')

plt.show()
# Visualising the Polynomial Regression Results

plt.scatter(x,y, color='red')

plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color='blue')

plt.title('Polynomial Regression')

plt.xlabel('Position')

plt.ylabel('Salary')

plt.show()