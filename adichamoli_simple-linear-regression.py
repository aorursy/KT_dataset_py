# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing the Dataset



dataset = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")

print(dataset.head())
print(dataset.dtypes)
X = dataset.iloc[:, :-1].values

y = dataset.iloc[:,1].values
# Splitting Dataset into Training and Test set



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
# Fitting Simple Linear Regression to Training Set



regressor = LinearRegression()

regressor.fit(X_train, y_train)
# Predicting Test Set Results



y_pred = regressor.predict(X_test)

print(y_pred)
# Mean Absolute Error



abs_err = mean_absolute_error(y_test, y_pred)

print(abs_err)
#Mean Squared Error



mse = mean_squared_error(y_test, y_pred)

print(mse)

print(np.sqrt(mse))
# Visualizing the Traing Set Results



plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary Vs Exp Training Set')

plt.xlabel('Years of Exp')

plt.ylabel('Salary')

plt.show()
# Visualizing the Test Set Results



plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title('Salary Vs Exp Test Set')

plt.xlabel('Years of Exp')

plt.ylabel('Salary')

plt.show()