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
df = pd.read_csv("/kaggle/input/world-happiness/2019.csv")
df.head(10)
print(df.isnull().sum())
x = df["GDP per capita"].values
print(x)
y = df["Score"].values
x = x.reshape(-1,1)
y = y.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_reg = LinearRegression()
lin_reg.fit(x_poly,y)
sorted_x = np.sort(x, axis=None).reshape(-1, 1)

y_pred = lin_reg.predict(poly_reg.fit_transform(sorted_x))
plt.scatter(x, y, color = 'red')
plt.plot(sorted_x, y_pred, color = 'blue')
plt.title('GDP vs Happiness Score')
plt.xlabel('GDP')
plt.ylabel('Happiness Score')
plt.show()
msqe = sum((y_pred - y) * (y_pred - y)) / y.shape[0]
rmse = np.sqrt(msqe)
print(msqe, rmse)