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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
pd.options.display.max_rows = 999
pd.options.display.max_columns = 999

data = pd.read_csv('/kaggle/input/szeged-weather/weatherHistory.csv')
data.head() 


data['Formatted Date'] = pd.to_datetime(data['Formatted Date'])
data = data.sort_values(by="Formatted Date")
col = data.columns
for c in col:
    print (c)
plt.scatter( np.log(data['Temperature (C)']), np.log(data['Humidity']))
plt.xlabel('Humidity')
plt.ylabel('Temperature')
plt.show()
plt.scatter(np.log(data['Apparent Temperature (C)']), np.log(data['Humidity']))
plt.xlabel('Apparent Temperature')
plt.ylabel('Humidity')
plt.show()
plt.scatter(np.log(data['Visibility (km)']), np.log(data['Humidity']))
plt.xlabel('Humidity')
plt.ylabel('Visibility in Km')
plt.show()
plt.hist((data['Humidity']))
plt.hist(data['Apparent Temperature (C)'])
plt.hist(data['Temperature (C)'])
print("Temperature", np.round(data['Temperature (C)'].isnull().mean(), 4),  ' % missing values')
print("Apparent Temperature", np.round(data['Apparent Temperature (C)'].isnull().mean(), 4),  ' % missing values')
print("Visibility (km)", np.round(data['Visibility (km)'].isnull().mean(), 4),  ' % missing values')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = data.iloc[:, 3:4].values
y = data.iloc[:, 5].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=True, random_state=0)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, lin_reg.predict(X_train), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()
y_pred = lin_reg.predict(X_test)
print("MSE = ",mean_squared_error(y_test, y_pred))
print("R2 = ",r2_score(y_test, y_pred))
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_train)
poly_reg.fit(X_poly, y_train)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y_train)
X_grid = np.arange(min(X_train), max(X_train), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.show()
y_pred = lin_reg_2.predict(poly_reg.fit_transform(X_test))
print("MSE = ",mean_squared_error(y_test, y_pred))
print("R2 = ",r2_score(y_test, y_pred))
lin_reg_2.predict(poly_reg.fit_transform([[40]]))
lin_reg.predict([[40]])
