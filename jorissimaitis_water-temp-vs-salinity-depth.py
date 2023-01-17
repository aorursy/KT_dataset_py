#Is there a relationship between water temperature and water salinity? - Joris Simaitis (Beginner)



#Importing Libraries

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 



#1.0 Getting dataset

"""First, I want to find out whether depth is correlated with water temperature. I predict yes, and if so, depth will have to be controlled. I will take a sample size of 500 to speed up analysis."""



data = pd.read_csv('../input/bottle.csv')

data = data[['Depthm', 'T_degC', 'Salnty']]

data = data.dropna(subset = ['T_degC', 'Depthm', 'Salnty'], how = 'any')

data = data[:][:500]

x = data.iloc[:, 0:1].values

y = data.iloc[:, 1].values



#1.1 Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



#1.2 PoLynomial Regression Fitting

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 3)

x_poly = poly_reg.fit_transform(x_train)

poly_reg.fit(x_poly, y_train)

lin_reg_2 = LinearRegression()

lin_reg_2.fit(x_poly, y_train)



#1.3 Data Visualisation

x_grid = np.arange(min(x_test), max(x_test), 0.1)

x_grid = x_grid.reshape((len(x_grid), 1))

plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)), color = 'blue')

plt.title('Effect of Depth on Water Temperature')

plt.xlabel('Depth')

plt.ylabel('Water Temperature')

plt.show()



"""A polynomial regression shows that indeed increasing in depth decreases the water temperature. Hence, I am going to analyse the relationship between water salinity/temperature for various depth conditions."""

#2.0 Depth at 0

data = pd.read_csv('../input/bottle.csv')

data = data[['Depthm', 'T_degC', 'Salnty']]

data = data.dropna(subset = ['T_degC', 'Depthm', 'Salnty'], how = 'any')

data = data[data.Depthm == 0]

data = data[:][:400]

x = data.iloc[:, 1:2].values

y = data.iloc[:, -1].values



#2.1 Training/test set split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



#2.2 Simple Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)

accuracy = lin_reg.score(x_test, y_test)

y_pred = lin_reg.predict(x_test)



#2.3 Data Visualisation

plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_test, y_pred, color = 'blue')

plt.title('Effect of Water Temperature on Water Salinity at Depth = 0')

plt.xlabel('Water Temperature')

plt.ylabel('Water Salinity')

plt.show()



#3.0 Depth below 10

data = pd.read_csv('../input/bottle.csv')

data = data[['Depthm', 'T_degC', 'Salnty']]

data = data.dropna(subset = ['T_degC', 'Depthm', 'Salnty'], how = 'any')

data = data[data.Depthm < 10]

data = data[:][:400]

x = data.iloc[:, 1:2].values

y = data.iloc[:, -1].values



#3.1 Training/test set split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



#3.2 Simple Linear Regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x_train, y_train)

accuracy = lin_reg.score(x_test, y_test)

y_pred = lin_reg.predict(x_test)



#3.3 Data Visualisation

plt.scatter(x_test, y_test, color = 'red')

plt.plot(x_test, y_pred, color = 'blue')

plt.title('Effect of Water Temperature on Water Salinity at Depth =< 10')

plt.xlabel('Water Temperature')

plt.ylabel('Water Salinity')

plt.show()



#4.0 Conclusion.
