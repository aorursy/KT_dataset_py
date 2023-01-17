# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import dates



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/covid19-global-forecasting-week-4/train.csv')

df_test = pd.read_csv('../input/covid19-global-forecasting-week-4/test.csv')



# df_train.head()

df_train.head()
def datesplit (df):

    year = []

    month = []

    day = []

    for item in df['Date']:

        x = item.split("-")

        year.append (x[0])    

        month.append (x[1])

        day.append (x[2])

        

    df['Year'] = year

    df['Month'] = month

    df['Day'] = day



datesplit (df_train)

datesplit (df_test)



df_train['ConfirmedCases'] = df_train['ConfirmedCases'].apply (int)

df_train['Fatalities'] = df_train['Fatalities'].apply (int)



df_train.head()



        
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



df_train['Province_State'].fillna('',inplace=True)

df_test['Province_State'].fillna('',inplace=True)





lbe = LabelEncoder()

df_train['Country_Region'] = lbe.fit_transform(df_train['Country_Region'])

df_test['Country_Region']  =  lbe.transform(df_test['Country_Region'])



df_train['Province_State'] = lbe.fit_transform(df_train['Province_State'])

df_test['Province_State'] =  lbe.transform(df_test['Province_State'])

    





    
df_train.head ()

from sklearn.preprocessing import MinMaxScaler

X_train = df_train.drop(["Id", "ConfirmedCases", "Fatalities", "Date"], axis = 1)

X_test = df_test.drop(["ForecastId","Date"], axis = 1)

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train.values)

X_test = scaler.transform(X_test.values)
X_train
y1 = df_train['ConfirmedCases']

y2 = df_train['Fatalities']

df_train.head()


y_train = y1

y_train_fat = y2
from xgboost import XGBRegressor

xgb = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)

y_pred = np.around(y_pred,decimals = 0)

y_pred
xgb1 = XGBRegressor(n_estimators = 1500 , random_state = 0 , max_depth = 15)

xgb1.fit(X_train,y_train_fat)
y_pred_fat = xgb1.predict(X_test)

y_pred_fat = np.around(y_pred_fat,decimals = 0)

y_pred_fat
df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

soln = pd.DataFrame({'ForecastId': df_test.ForecastId, 'ConfirmedCases': y_pred, 'Fatalities': y_pred_fat})

df_out = pd.concat([df_out, soln], axis=0)

df_out.ForecastId = df_out.ForecastId.astype('int')

df_out.ConfirmedCases = df_out.ConfirmedCases.astype('int')

df_out.Fatalities = df_out.Fatalities.astype('int')

df_out.to_csv('submission.csv', index=False)