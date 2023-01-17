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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/Position_Salaries.csv')
dataset
X = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values
X, y
from sklearn.linear_model import LinearRegression

lin_regressor = LinearRegression()

lin_regressor.fit(X,y)
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree = 4)

X_poly = poly_reg.fit_transform(X)
X_poly
lin_regressor2 = LinearRegression()

lin_regressor2.fit(X_poly, y)
# Visualising the Linear Regression result

plt.scatter(X,y, color='red')

plt.plot(X, lin_regressor.predict(X), color='blue')

plt.title("Truth or Bluff (Linear Regression)")

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.show()
# Vsualising the Polynomial Regression result

plt.scatter(X,y, color='red')

plt.plot(X, lin_regressor2.predict(poly_reg.fit_transform(X)), color='blue')

plt.title("Truth or Bluff (Polynomial Regression)")

plt.xlabel('Position Level')

plt.ylabel('Salary')

plt.show()
# Predicting a new result with Linear Regression

# y_pred = lin_regressor.predict(5.5)
# Predicting a new result with Polynomial Regression

# lin_regressor2.predict(poly_reg.fit_transform(6))