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
df = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")
df.head(50)
df.tail(50)
df.describe()
df.info()
#t1- fix Nulls - Province_State Feature

df[pd.isnull(df["Province_State"])]

df.Province_State=df.Province_State.fillna(df.Country_Region)

#t2-all data in num



df.drop(['Id'], inplace = True, axis = 1)

df.info()

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



df['Province_State'] = le.fit_transform(df['Province_State'])

df['Country_Region'] = le.fit_transform(df['Country_Region'])

df.info()
#Handle Date Feature Create new features from date



# Convert Date feature to datetime

df.Date = df.Date.apply(pd.to_datetime)



# Create new features from Date

df['Day_Of_The_Year'] = df['Date'].dt.day

df['Month_Of_The_Year'] = df['Date'].dt.month

df['Week_Of_The_Year'] = df['Date'].dt.week

df.drop(['Date'], inplace = True, axis = 1)

df.info()
#X and y

# Separate X and Y but there will be 2 Y's

X = df.drop(['ConfirmedCases', 'Fatalities'], axis = 1)

y_cc = df['ConfirmedCases']

y_fat = df['Fatalities']

           
from sklearn.model_selection import train_test_split

X_train_cc, X_test_cc, y_train_cc, y_test_cc = train_test_split(X, y_cc, test_size=0.3)

X_train_fat, X_test_fat, y_train_fat, y_test_fat = train_test_split(X, y_fat, test_size=0.3)
# Train Regressor - Linear Regresion

from sklearn.tree import DecisionTreeRegressor



# Confirmed Cases



reg_dt_cc = DecisionTreeRegressor()

reg_dt_cc.fit(X_train_cc, y_train_cc)

y_pred_cc = reg_dt_cc.predict(X_test_cc)



# Fatalities



reg_dt_fat = DecisionTreeRegressor()

reg_dt_fat.fit(X_train_fat, y_train_fat)

y_pred_fat = reg_dt_fat.predict(X_test_fat)

import math

from sklearn.metrics import mean_squared_log_error

rmsle_cc = math.sqrt(mean_squared_log_error(y_test_cc, y_pred_cc))

rmsle_fat = math.sqrt(mean_squared_log_error(y_test_fat, y_pred_fat))

print(rmsle_cc)

print(rmsle_fat)
# Train Regressor - Lasso Regresion

from sklearn import linear_model



reg_lasso_cc = linear_model.Lasso(alpha=0.1)

reg_lasso_cc.fit(X_train_cc, y_train_cc)

y_pred_cc = reg_lasso_cc.predict(X_test_cc)



reg_lasso_fat = linear_model.Lasso(alpha=0.1)

reg_lasso_fat.fit(X_train_fat, y_train_fat)

y_pred_fat = reg_lasso_fat.predict(X_test_fat)



import math

from sklearn.metrics import mean_squared_log_error

rmsle_cc = math.sqrt(mean_squared_log_error(y_test_cc, y_pred_cc))

rmsle_fat = math.sqrt(mean_squared_log_error(y_test_fat, y_pred_fat))

print(rmsle_cc)

print(rmsle_fat)