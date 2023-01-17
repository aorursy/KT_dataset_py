from xgboost import XGBRegressor# This Python 3 environment comes with many helpful analytics libraries installed
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
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')
df_train.head()
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
df_test.head()
df_train.info()
df_test.info()
df_train.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df_train['Country/Region'])
df_train['Country/Region'] = le.transform(df_train['Country/Region'])
df_test['Country/Region'] = le.transform(df_test['Country/Region'])
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train['Days_passed'] = (df_train['Date'] - df_train['Date'].min()).dt.days
df_test['Date'] = pd.to_datetime(df_test['Date'])
df_test['Days_passed'] = (df_test['Date'] - df_train['Date'].min()).dt.days
df_train.info()
df_test.info()
df_train.head()
df_test.head()
df_train2 = df_train.drop(['Id','Province/State','Date','Fatalities'], axis=1)
df_train2.head()
X_train = df_train2.drop('ConfirmedCases', axis=1)
y_train = df_train2['ConfirmedCases']
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
xg = XGBRegressor()

parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid = GridSearchCV(xg,
                    parameters, n_jobs=4,
                    #scoring="neg_log_loss",
                    cv=3)


grid.fit(X_train, y_train)
df_test2 = df_test.drop(['ForecastId','Province/State','Date'], axis=1)
X_test = df_test2
df_test['ConfirmedCases'] = grid.predict(X_test)
df_train3 = df_train.drop(['Id','Province/State','Date'], axis=1)
df_train3.head()
X_train2 = df_train3.drop('Fatalities', axis=1)
y_train2 = df_train3['Fatalities']
xg2 = XGBRegressor()

parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid2 = GridSearchCV(xg2,
                    parameters, n_jobs=-1,
                    #scoring="neg_log_loss",
                    cv=3)

grid2.fit(X_train2, y_train2)
df_test3 = df_test.drop(['ForecastId','Province/State','Date'], axis=1)
df_test3.head()
X_test2 = df_test3[['Country/Region', 'Lat', 'Long', 'ConfirmedCases', 'Days_passed']]
df_test['Fatalities'] = grid2.predict(X_test2)
df_test.head()
output = df_test[['ForecastId','ConfirmedCases','Fatalities']]
output.head()
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")
