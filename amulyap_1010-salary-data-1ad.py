import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import sklearn.metrics as sm

#Amulya lab task1
dataset = pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')

X = dataset[['YearsExperience']].to_numpy()

y = dataset[['Salary']].to_numpy()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=5)#the value of test_size has to be changed according to the question

#The shape for test and train are printed for analysis

print(X_train.shape)

print(X_test.shape)

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

def mse_func(y_test,y_pred):

  s = (y_test-y_pred)**2

  s = np.sum(s)

  s = s/len(y_test)

  return (s)

def rmse_func(y_test,y_pred):

  return (mse_func(y_pred,y_test))**0.5

plt.figure()

plt.scatter(X_test,y_test)

plt.plot(X_test,y_pred,color='black',linewidth=4)

plt.xticks(())

plt.yticks(())

plt.show()

#lab task amulya_ponnuru

print('Mean Square error: ',round(sm.mean_squared_error(y_test,y_pred,squared=True),2))

print('Root Mean Square error: ',round(sm.mean_squared_error(y_test,y_pred,squared=False),2))

print('MSE calculated: ',mse_func(y_test,y_pred))

print('RMSE calculated: ',rmse_func(y_test,y_pred))
