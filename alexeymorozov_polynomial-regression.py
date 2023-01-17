# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression

from pylab import *

from sklearn.metrics import r2_score

count = 100
dataset = pd.read_csv('../input/california-housing-prices/housing.csv')

x = dataset.iloc[0:count, 2:3].values

y = dataset.iloc[0:count, 8].values

scatter(x, y)
poly_reg = PolynomialFeatures(degree = 6)

x_poly = poly_reg.fit_transform(x)

pol_reg = LinearRegression()

pol_reg.fit(x_poly, y)

print('Coefficients = ', pol_reg.coef_)
x_grid = np.arange(min(x), max(x), 1 / 2)

x_grid = x_grid.reshape((len(x_grid), 1))

y_grid = pol_reg.predict(poly_reg.fit_transform(x_grid))

plt.scatter(x, y, color = 'red')

plt.plot(x_grid, y_grid, color = 'blue')

plt.title('Total_bedrooms dependence on the total_rooms of the house in the quarter')

plt.xlabel('Total rooms')

plt.ylabel('Total bedrooms')

plt.show()
pol_reg.predict(poly_reg.fit_transform([[100]]))
r2 = r2_score(y, y_grid)

print(r2)