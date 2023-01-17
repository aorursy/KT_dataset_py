# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/Position_Salaries.csv")
df.head()
# Following plot the relationship between Level and Salary

plt.figure(figsize=(15, 10))

plt.scatter(x=df["Level"], y=df["Salary"])

plt.xlabel("Level");

plt.ylabel("Salary")
X = df.iloc[:, 1:2].values
y = df.iloc[:, -1:].values
X
# We will create both the models, So that we can compare th prediction.

# As we know we have very less number of observations so we will not split our data into train and test 

# Though if you have big data, you can do it using train_test_split class.
# lets import model class

from sklearn.linear_model import LinearRegression
# linear regression model

lr_normal = LinearRegression()

lr_normal.fit(X, y)
# import Polynomial Features

from sklearn.preprocessing import PolynomialFeatures
# polynomial regression model

# degree 2 means it will add one more column to your dataset with X to the power 2. X^2.

# You will observer, there is one more column on X_poly at the 0th position(1st column) which has value 1 for

# all rows. It is added by PolynomialFeatures. It is the intercept value 'c' in Y=c+ m1*x + m2*x^2

lr_poly = LinearRegression()

poly_reg = PolynomialFeatures(degree = 2) 

X_poly = poly_reg.fit_transform(X)
X_poly
lr_poly.fit(X_poly, y)
# Visualising the Linear Regression results

plt.figure(figsize=(15, 10))

plt.scatter(X, y, color = 'red')

plt.plot(X, lr_normal.predict(X), color = 'blue')

plt.title('Truth or Bluff (Linear Regression)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()



# Red is actual Salaries and blue line is predicted salaries.

# So here we can see that best fitted line is also not a good regressor for the output. You can compare that

# Mostly Salaries are very far from actual values. Now lets see what our polynomial model shows


# Visualising the Polynomial Regression results

plt.figure(figsize=(15, 10))

plt.scatter(X, y, color = 'red')

plt.plot(X, lr_poly.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Position Level Vs Salary (Polynomial Regression-Degree 2)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()



# So we can see our polynomial model of degree 2 is very simillar to actual result.

# lets try for higher degree polynomial. Very simple to do, We just have to change the degree parameter

# of PolynomialFeatures() constructor

# polynomial regression model

lr_poly = LinearRegression()

poly_reg = PolynomialFeatures(degree = 3) # change form 2 to 3

X_poly = poly_reg.fit_transform(X)

lr_poly.fit(X_poly, y)
# Visualising the Polynomial Regression results

plt.figure(figsize=(15, 10))

plt.scatter(X, y, color = 'red')

plt.plot(X, lr_poly.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Position Level Vs Salary (Polynomial Regression- Degree 3)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()



# So now you can see as we increaed our degree parameter from 2 to 3, salary predictions are even better.

# Lets just check for 4
# So we can see our polynomial model of degree 2 is very simillar to actual result.

# lets try for higher degree polynomial. Very simple to do, We just have to change the degree parameter

# of PolynomialFeatures() constructor

# polynomial regression model

lr_poly = LinearRegression()

poly_reg = PolynomialFeatures(degree = 4) # change form  3 to 4

X_poly = poly_reg.fit_transform(X)

lr_poly.fit(X_poly, y)
# Visualising the Polynomial Regression results

plt.figure(figsize=(15, 10))

plt.scatter(X, y, color = 'red')

plt.plot(X, lr_poly.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Position Level Vs Salary  (Polynomial Regression-Degree 4)')

plt.xlabel('Position level')

plt.ylabel('Salary')

plt.show()

# Wow!! so now our model is even more accurate than polynomial curve with degree 3.
## So here we can see as we increased the degree of polynomial, our curve acuracy increased. So you should

# try different degree to find out best fitted curve.

# Now lets predict some values

lr_normal.predict([[8.5]]) # As per our linear regresion model salary.of 8.5 level should be 492136.

# but according to curve it should be somewhere lesser than 400000.

# lets see what our polynomial model says
lr_poly.predict(poly_reg.fit_transform([[8.5]]))  # According to this it should be 387705

# if you will see the figure you will find this prediction is almost same as what we are expecting.
## Thank You