# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Importing the libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#Importing the dataset 



dataset= pd.read_csv('../input/Position_Salaries.csv')



dataset.head(6)
#Assiging the dataset into X and y varialbles
X = dataset.iloc[:, 1:2]

y = dataset.iloc[:, 2:]



print(X.head())

print(y.head())
#Fitting the linear Regression model into the dataset
from sklearn.linear_model import LinearRegression



regressor_LR = LinearRegression().fit(X, y)



y_predict = regressor_LR.predict(X)



y_predict
#Visualising the dataset using the Linear Rergression o/p
plt.scatter(X , y, color = 'red')

plt.plot(X, regressor_LR.predict(X), color = 'blue')

plt.title('Positon Salary')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
#Fitting the Polynomial Regression into the dataset
from sklearn.preprocessing import PolynomialFeatures



poly_reg = PolynomialFeatures(degree = 4)



X_poly = poly_reg.fit_transform(X)



X_poly
#Now fitting the regression model using the X_poly
regression_PR = LinearRegression().fit(X_poly, y.values)



y_predict_PR= regression_PR.predict(poly_reg.fit_transform(X))



y_predict_PR
#Visualising the dataset using the Polynomial Rergression o/p
plt.scatter(X , y, color = 'red')

plt.plot(X, regression_PR.predict(poly_reg.fit_transform(X)), color = 'blue')

plt.title('Positon Salary')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()
#Visualising the polynomial regression model using higer definition
X_new = X.values
X_grid = np.arange(min(X_new), max(X_new), 0.1)

X_df = pd.DataFrame(X_grid)

X_df.head()

X_df.shape


plt.scatter(X , y, color = 'red')

plt.plot(X_df, regression_PR.predict(poly_reg.fit_transform(X_df)), color = 'purple')

plt.title('Positon Salary')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.show()