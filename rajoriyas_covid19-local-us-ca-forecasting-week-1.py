#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Sat Mar 21 15:36:47 2020



@author: rahul

"""



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import datetime



df_train=pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df_test=pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

df_sub=pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_submission.csv')



"""

df_train=pd.read_csv('ca_train.csv')

df_test=pd.read_csv('ca_test.csv')

df_sub=pd.read_csv('ca_submission.csv')

"""



reported=df_train[df_train['Date']>='2020-03-10'].reset_index()

reported['day_count']=list(range(1, len(reported)+1))



X_train=reported.iloc[:, 9:10].values

y_train=reported.iloc[:, 7:8].values



X_test=list(range(1, 53))



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

poly_reg=PolynomialFeatures(degree=2)

X_poly=poly_reg.fit_transform(X_train)

lin_reg=LinearRegression()

lin_reg.fit(X_poly, y_train)



X_grid = np.arange(min(X_train), max(X_train), 0.1)

X_grid = X_grid.reshape((len(X_grid), 1))



plt.scatter(X_train, y_train, color = 'red')

plt.scatter(X_train, y_train, color='red')

plt.plot(X_grid, lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')

plt.title('Covid 19')

plt.xlabel('Date')

plt.ylabel('Confirmed Cases-Training Dataset')

plt.show()



y_train_pred=lin_reg.predict(poly_reg.fit_transform(X_train))



X_test=np.arange(1, 53, 1)

X_test = X_test.reshape((len(X_test), 1))



y_pred=lin_reg.predict(poly_reg.fit_transform(X_test))



plt.plot(X_test, y_pred, color='blue')

plt.title('Covid 19')

plt.xlabel('Date')

plt.ylabel('Confirmed Cases-Test Dataset')

plt.show()



regressor=LinearRegression()

X_train_2=reported[['ConfirmedCases']].values

y_train_2=reported[['Fatalities']].values

regressor.fit(X_train_2, y_train_2)



X_test_2=y_pred

y_pred_2 = regressor.predict(X_test_2)



plt.scatter(X_train_2, y_train_2, c='blue')

plt.plot(X_test_2, y_pred_2, c='red')

plt.title("Regression Line")

plt.xlabel('ConfirmedCases')

plt.ylabel('Fatalities')

plt.grid()

plt.xlim([100,800])

plt.ylim([0,20])

plt.show()



result=pd.DataFrame(y_pred)

result[1]=y_pred_2

submission=result[0:len(df_sub)].reset_index()

df_sub['ConfirmedCases'] = submission[0]

df_sub['Fatalities'] = submission[1]



df_sub.to_csv("submission.csv", index=False)