#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Wed Mar 25 11:25:25 2020



@author: rahul

"""

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime



df_train=pd.read_csv('../input/covid19-global-forecasting-week-1/train.csv')

df_test=pd.read_csv('../input/covid19-global-forecasting-week-1/test.csv')

df_sub=pd.read_csv('../input/covid19-global-forecasting-week-1/submission.csv')

"""

df_train=pd.read_csv('train.csv')

df_test=pd.read_csv('test.csv')

df_sub=pd.read_csv('submission.csv')

""" 



df_train.info()



df_train["Date"] = df_train["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

df_train["Date"] = df_train["Date"].apply(lambda x: x.timestamp())

df_train["Date"] = df_train["Date"].astype(int)



#df_train['Day']=list(range(1, len(df_train)+1))



#df_train.drop(['ForecastId', 'Province/State', 'Country/Region'], axis=1, inplace=True)

df_train=df_train[['Date', 'Lat', 'Long', 'ConfirmedCases', 'Fatalities']]

df_train.info()



X_train=df_train.iloc[:, :-2].values

y_ConfirmedCases_train=df_train.iloc[:, 3:4].values

y_Fatalities_train=df_train.iloc[:, 4:5].values



df_test["Date"] = df_test["Date"].apply(lambda x: datetime.strptime(x,'%Y-%m-%d'))

df_test["Date"] = df_test["Date"].apply(lambda x: x.timestamp())

df_test["Date"] = df_test["Date"].astype(int)



#df_test['Day']=list(range(1, len(df_test)+1))



#df_test.drop(['ForecastId', 'Province/State', 'Country/Region'], axis=1, inplace=True)

df_test=df_test[['Date', 'Lat', 'Long']]

df_test.info()

X_test=df_test.iloc[:, :].values



from sklearn.ensemble import RandomForestRegressor

reg=RandomForestRegressor(bootstrap=True,max_depth=None, max_features='auto', max_leaf_nodes=None, n_estimators=250, random_state=None, n_jobs=1, verbose=0)

reg.fit(X_train, y_ConfirmedCases_train)



y_ConfirmedCases_train_pred=reg.predict(X_train)

y_ConfirmedCases_pred=reg.predict(X_test)



reg_2=RandomForestRegressor(bootstrap=True,max_depth=None, max_features='auto', max_leaf_nodes=None, n_estimators=250, random_state=None, n_jobs=1, verbose=0)

reg_2.fit(X_train, y_Fatalities_train)



y_Fatalities_train_pred=reg.predict(X_train)

y_Fatalities_pred=reg.predict(X_test)



df_sub['ConfirmedCases']=y_ConfirmedCases_pred

df_sub['Fatalities']=y_Fatalities_pred

df_sub["ConfirmedCases"] = df_sub["ConfirmedCases"].astype(int)

df_sub["Fatalities"] = df_sub["Fatalities"].astype(int)

df_sub.to_csv("submission.csv", index=False)
