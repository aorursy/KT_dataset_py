# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Reading data-set
carPrice = pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
print(carPrice.shape)
print(carPrice.head(5))
carPrice.describe()
# encoding non-int and non-float to integer values
from sklearn.preprocessing import LabelEncoder
lab=LabelEncoder()
data = carPrice
data['fuelsystem']=lab.fit_transform(data['fuelsystem'])
data['cylindernumber']=lab.fit_transform(data['cylindernumber'])
data['enginetype']=lab.fit_transform(data['enginetype'])
data['enginelocation']=lab.fit_transform(data['enginelocation'])
data['drivewheel']=lab.fit_transform(data['drivewheel'])
data['carbody']=lab.fit_transform(data['carbody'])
data['doornumber']=lab.fit_transform(data['doornumber'])
data['aspiration']=lab.fit_transform(data['aspiration'])
data['fueltype']=lab.fit_transform(data['fueltype'])
data['CarName']=lab.fit_transform(data['CarName'])
data.head(5)
data.info()
x = data.iloc[:,1:25]
y = data.iloc[:,25]
# Multiple Linear Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
from sklearn.metrics import r2_score
y_pred = regressor.predict(x_test)
print(r2_score(y_test,y_pred))
df = pd.DataFrame({'Actual':y_test.values, 'Predicted':y_pred})
ax1 = df.plot.scatter(x='Actual', y='Predicted')
import matplotlib.pyplot as plt
x_poly = x['enginesize']
x_poly = np.array(x_poly)
x_poly = x_poly.reshape(-1,1)
y_poly = y
y_poly = np.array(y_poly)
plt.scatter(x_poly, y_poly, color = 'red')
# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
x_train, x_test, y_train, y_test = train_test_split(x_poly, y_poly, test_size=0.2, random_state=0)
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(x_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
y_pred = lin_reg_2.predict(X_poly)
print(r2_score(y_train, y_pred))
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_test, lin_reg_2.predict(poly_reg.fit_transform(x_test)), color = 'blue')
plt.show()
