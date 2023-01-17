import pickle

import numpy as np

from sklearn import linear_model

import sklearn.metrics as sm

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import preprocessing,svm

from sklearn.linear_model import LinearRegression

import pandas as pd
data = pd.read_csv ('../input/salary-datacsv/salary_data.csv')

print (data)

print(data.head())
X = np.array(data['YearsExperience']).reshape(-1, 1) 

y = np.array(data['Salary']).reshape(-1, 1)
from sklearn.model_selection import train_test_split

regressor = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.8,test_size = 0.2,random_state=0)

X_train.shape
X_test.shape
regressor.fit(X_train, y_train)

y_test_pred=regressor.predict(X_test)

print(y_test_pred)
plt.scatter(X_test, y_test, color='red')

plt.plot(X_test, y_test_pred, color='black', linewidth=2)

plt.title('Salary vs Years of Experience (Test Data 20%)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.xticks(())

plt.yticks(())

plt.show()
X_test_pred = regressor.predict(X_train)

X_test_pred

plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color='black')

plt.title('Salary vs Years of Experience(Train data 80%)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.xticks(())

plt.yticks(())

plt.show()

import math

y_actual=np.array(y_test)

y_pred=np.array(y_test_pred)

error=(y_actual-y_pred)**2

error_mean=(np.mean(error))

err_sq=math.sqrt(error_mean)

err_sq

mse=error_mean

rmse=err_sq
error_mean
print("mean square error is" ,mse)
print("root mean square error is" ,rmse)
import math

print("Linear regressor performance:")

print("root Mean squared error =", math.sqrt(sm.mean_squared_error(y_test, y_test_pred)))

print("Mean squared error =", (sm.mean_squared_error(y_test, y_test_pred)))
from sklearn.model_selection import train_test_split

regressor = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.5,test_size = 0.5,random_state=0)

X_train.shape
X_test.shape

regressor.fit(X_train, y_train)

y_test_pred=regressor.predict(X_test)

print(y_test_pred)

plt.scatter(X_test, y_test, color='red')

plt.plot(X_test, y_test_pred, color='black', linewidth=2)

plt.title('Salary vs Years of Experience (Test Data 50%)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.xticks(())

plt.yticks(())

plt.show()

X_test_pred = regressor.predict(X_train)

X_test_pred

plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color='black')

plt.title('Salary vs Years of Experience(Train data 50%)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.xticks(())

plt.yticks(())

plt.show()

import math

y_actual=np.array(y_test)

y_pred=np.array(y_test_pred)

error=(y_actual-y_pred)**2

error_mean=(np.mean(error))

err_sq=math.sqrt(error_mean)

err_sq

mse=error_mean

rmse=err_sq

print("mean square error is" ,mse)

print("root mean square error is" ,rmse)

import math

print("Linear regressor performance:")

print("root Mean squared error =", math.sqrt(sm.mean_squared_error(y_test, y_test_pred)))

print("Mean squared error =", (sm.mean_squared_error(y_test, y_test_pred)))
from sklearn.model_selection import train_test_split

regressor = linear_model.LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y,train_size = 0.7,test_size = 0.3,random_state=0)

X_train.shape
X_test.shape

regressor.fit(X_train, y_train)

y_test_pred=regressor.predict(X_test)

print(y_test_pred)

plt.scatter(X_test, y_test, color='red')

plt.plot(X_test, y_test_pred, color='black', linewidth=2)

plt.title('Salary vs Years of Experience (Test Data 30%)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.xticks(())

plt.yticks(())

plt.show()

X_test_pred = regressor.predict(X_train)

X_test_pred

plt.scatter(X_train,y_train,color='red')

plt.plot(X_train,regressor.predict(X_train),color='black')

plt.title('Salary vs Years of Experience(Train data 70%)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary')

plt.xticks(())

plt.yticks(())

plt.show()

import math

y_actual=np.array(y_test)

y_pred=np.array(y_test_pred)

error=(y_actual-y_pred)**2

error_mean=(np.mean(error))

err_sq=math.sqrt(error_mean)

err_sq

mse=error_mean

rmse=err_sq

print("mean square error is" ,mse)

print("root mean square error is" ,rmse)

import math

print("Linear regressor performance:")

print("root Mean squared error =", math.sqrt(sm.mean_squared_error(y_test, y_test_pred)))

print("Mean squared error =", (sm.mean_squared_error(y_test, y_test_pred)))