import numpy as np # for linear algebra

import pandas as pd # data processing, CSV file I/O

data = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')

data.head()
def mse(predictions, targets):

    return  (((predictions - targets) ** 2).mean())
def rmse(predictions, targets):

    return np.sqrt(((predictions - targets) ** 2).mean())
x = data.iloc[:, :-1] # YearsExperience as x

y = data.iloc[:, [-1]] # Salary as y
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.50,test_size = 0.50) # 50% Train and 50% Test
from sklearn import linear_model

lr = linear_model.LinearRegression()

lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
print('Mean squared error using fuction: %.2f' % mse(y_test,y_pred))

print('Root mean squared error using fuction: %.2f' % rmse(y_test, y_pred))
from sklearn.metrics import mean_squared_error

print('Mean squared error using sklearn: %.2f' % mean_squared_error(y_test,y_pred))

rms = np.sqrt(mean_squared_error(y_test, y_pred))

print('Root mean squared error using sklearn: %.2f' % rms)
x1 = data.iloc[:, :-1] # YearsExperience as x

y1 = data.iloc[:, [-1]] # Salary as y
from sklearn.model_selection import train_test_split

x1_train,x1_test,y1_train,y1_test = train_test_split(x1,y1,train_size = 0.70,test_size = 0.30) # 70% Train and 30% Test
from sklearn import linear_model

lr = linear_model.LinearRegression()

lr.fit(x1_train,y1_train)

y1_pred = lr.predict(x1_test)
print('Mean squared error using fuction: %.2f' % mse(y1_test,y1_pred))

print('Root mean squared error using fuction: %.2f' % rmse(y1_test, y1_pred))
from sklearn.metrics import mean_squared_error

print('Mean squared error using sklearn: %.2f' % mean_squared_error(y1_test,y1_pred))

rms = np.sqrt(mean_squared_error(y1_test, y1_pred))

print('Root mean squared error using sklearn: %.2f' % rms)
x2 = data.iloc[:, :-1] # YearsExperience as x

y2 = data.iloc[:, [-1]] # Salary as y
from sklearn.model_selection import train_test_split

x2_train,x2_test,y2_train,y2_test = train_test_split(x2,y2, train_size = 0.80,test_size = 0.2) # 80% Train and 20% Test
from sklearn import linear_model

lr = linear_model.LinearRegression()

lr.fit(x2_train,y2_train)

y2_pred = lr.predict(x2_test)
print('Mean squared error using fuction: %.2f' % mse(y2_test,y2_pred))

print('Root mean squared error using fuction: %.2f' % rmse(y2_test, y2_pred))
from sklearn.metrics import mean_squared_error

print('Mean squared error using sklearn: %.2f' % mean_squared_error(y2_test,y2_pred))

rms = np.sqrt(mean_squared_error(y2_test, y2_pred))

print('Root mean squared error using sklearn: %.2f' % rms)