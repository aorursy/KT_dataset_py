#Necessary Imports for importing the required modules to be used

import pandas as pd

import numpy  as np

import matplotlib.pyplot as plt

%matplotlib inline   

# this makes sure that the graphs are printed in the jupyter notebook itself
#importing the dataset

dataset= pd.read_csv('../input/polynomial-regression-salaries/Position_Salaries.csv') 

dataset.head()
x=dataset.iloc[:,1:2].values

x
y=dataset.iloc[:,2].values

y
# Fitting Linear Regression to the dataset

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(x, y)
plt.scatter(x, y, color = 'red')

plt.plot(x, lin_reg.predict(x), color = 'blue')

plt.title('Linear Regression')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
# Fitting Polynomial Regression to the dataset

from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 2)  #trying to create a 2 degree polynomial equation. It simply squares the x as shown in the output

X_poly = poly_reg.fit_transform(x)

print(X_poly)

poly_reg.fit(X_poly, y)
# doing the actual polynomial Regression

lin_reg_2 = LinearRegression()

lin_reg_2.fit(X_poly, y)
# Visualising the Polynomial Regression results

plt.scatter(x, y, color = 'red')

plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')

plt.title('Polynomial Regression')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
# Fitting Polynomial Regression to the dataset

poly_reg1 = PolynomialFeatures(degree = 4)

X_poly1 = poly_reg1.fit_transform(x)

poly_reg1.fit(X_poly, y)

lin_reg_3 = LinearRegression()

lin_reg_3.fit(X_poly1, y)
# Visualising the Polynomial Regression results

plt.scatter(x, y, color = 'red')

plt.plot(x, lin_reg_3.predict(poly_reg1.fit_transform(x)), color = 'blue')

plt.title('Polynomial Regression of Degree 4')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()
# Now the question, did you find anything odd in our model????