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
import seaborn as sns

import matplotlib.pyplot as plt
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

test =  pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/test.csv')
train = train.drop(['Province/State'], axis=1)

test = test.drop(['Province/State'], axis=1)

print ('train shape:', train.shape)

print ('train missing: ', train.isnull().sum())

print ('test shape: ', test.shape)

print ('test missing: ', test.isnull().sum())
print (train.columns)

print (test.columns)
print (train.isnull().sum())

print ('*'*100)

print (test.isnull().sum())
print (train.head())

print ('*'*100)

print (test.head())
print (train.Date.unique())
print (train['Country/Region'].unique())
print (len(train['Id'].unique().tolist()))

print (len(test['ForecastId'].unique().tolist()))
train_cases_confirmed = train.groupby('Date')['ConfirmedCases'].sum()

train_cases_fatal = train.groupby('Date')['Fatalities'].sum()

print (train_cases_confirmed.head())

train_cases_confirmed.plot()

# train_cases_fatal.plot()
train_cases_confirmed_china = train[train['Country/Region']=='China'].groupby('Date')['ConfirmedCases'].sum()

train_cases_fatal_china = train[train['Country/Region']=='China'].groupby('Date')['Fatalities'].sum()

print (train_cases_confirmed_china.head())

train_cases_confirmed_china.plot()

# train_cases_fatal.plot()
train_case_by_country = train.groupby(['Country/Region'], as_index=False)['ConfirmedCases'].sum()

# train_case_by_country = train.groupby(['Country/Region'], as_index=False).agg({'ConfirmedCases':['sum']})

print (train_case_by_country.head())



sorted_train_case_by_country = train_case_by_country.sort_values('ConfirmedCases', ascending=False)

print (sorted_train_case_by_country)



plt.bar(sorted_train_case_by_country['Country/Region'][:5], sorted_train_case_by_country['ConfirmedCases'][:5], 

        color=['red','yellow','black','blue','green'])
train_country_date = train.groupby(['Country/Region', 'Date', 'Lat', 'Long'], as_index=False)['ConfirmedCases', 'Fatalities'].sum()

print (train_country_date.tail(100))

print (train_country_date.shape)
print (train_country_date.info())

print (train_country_date.isnull().sum())
train_country_date.Date = pd.to_datetime(train_country_date['Date'])

print (train_country_date.head())

print (train_country_date.info())
train_country_date['month']=train_country_date['Date'].dt.month

train_country_date['day']=train_country_date["Date"].dt.day

train_country_date['dayofweek']=train_country_date['Date'].dt.dayofweek

print (train_country_date.head(10))

print (train_country_date.describe())

print (train_country_date.columns)
print (train_country_date['Date'].min)

print (train_country_date.isnull().sum())
print (train_country_date.shape)

print (train_country_date.columns)
test.Date = pd.to_datetime(test['Date'])

test['month']=test['Date'].dt.month

test['day']=test["Date"].dt.day

test['dayofweek']=test['Date'].dt.dayofweek



print (test.head(10))

print (test.describe())

print (test.columns)

print (test.shape)
# print (train_test.shape)

# print (train_test.head())
pre_columns =['Country/Region','Lat','Long', 'month', 'day','dayofweek']

train_country_date_select = train_country_date[pre_columns]

test_select = test[pre_columns]

train_test = pd.concat([train_country_date_select, test_select], axis=0)
from sklearn.preprocessing import LabelEncoder

from category_encoders.ordinal import OrdinalEncoder

from category_encoders.woe import WOEEncoder

from category_encoders.target_encoder import TargetEncoder

from category_encoders.sum_coding import SumEncoder

from category_encoders.m_estimate import MEstimateEncoder

from category_encoders.leave_one_out import LeaveOneOutEncoder

from category_encoders.helmert import HelmertEncoder

from category_encoders.cat_boost import CatBoostEncoder

from category_encoders.james_stein import JamesSteinEncoder

from category_encoders.one_hot import OneHotEncoder



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
label_encoder = LabelEncoder()

train_test['country'] = label_encoder.fit_transform(train_test['Country/Region'])

print (train_test.tail())
columns = ['country', 'Lat','Long', 'month', 'day', 'dayofweek']

train_test_clean = train_test[columns]

print (train_test_clean.shape)

print (train_test_clean.head())
train_data = train_test_clean[:17892]

print (train_data.shape)

print (train_data.columns)

print (train_case_by_country.shape)
train_data_clean= pd.concat([train_data,train_country_date['ConfirmedCases'], train_country_date['Fatalities']], axis=1)

print (train_data_clean.shape)

print (train_data_clean.columns)

print (train_data_clean.head())
test_data_clean = train_test_clean[17892:]

print (test_data_clean.shape)

print (test_data_clean.columns)

print (test_data_clean.head())

print (test_data_clean.isnull().sum())
train_data_clean = train_data_clean.dropna()


columns =['country', 'Lat', 'Long', 'month', 'day', 'dayofweek']

X = train_data_clean[columns]

Y_cases = train_data_clean['ConfirmedCases']

Y_fatal = train_data_clean['Fatalities']
print (X.isnull().sum())

# X = X.dropna()

# print (X.isnull().sum())

# print (X['Lat'].unique())

# print (Y_cases.isnull().sum())
X_train_case, X_val_case, y_train_case, y_val_case = train_test_split(X, Y_cases, test_size=0.3, random_state=42)
rf = RandomForestRegressor()

rf.fit(X_train_case, y_train_case)

y_predicted_cases_val = rf.predict(X_val_case)
from sklearn.metrics import mean_squared_error

print (y_predicted_cases_val)

mse_cases_val = mean_squared_error(y_val_case, y_predicted_cases_val)

print (mse_cases_val)
y_predicted_cases_test = rf.predict(test_data_clean)

print (y_predicted_cases_test.shape)

print (y_predicted_cases_test[:5])
X_train_fatal, X_val_fatal, y_train_fatal, y_val_fatal = train_test_split(X, Y_fatal, test_size=0.3, random_state=42)
rf_fatal = RandomForestRegressor()

rf_fatal.fit(X_train_fatal, y_train_fatal)

y_predicted_fatal_val = rf_fatal.predict(X_val_fatal)
mse_fatal_val = mean_squared_error(y_val_fatal, y_predicted_fatal_val)

print (mse_fatal_val)
y_predicted_fatal_test = rf_fatal.predict(test_data_clean)

print (y_predicted_fatal_test.shape)

print (y_predicted_fatal_test[:5])
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/submission.csv')
print (submission.shape)

print (submission.head())
submission.drop(['ConfirmedCases','Fatalities'], axis=1, inplace=True)

print (submission.head())

print (submission.shape)
print (y_predicted_cases_test.shape)

print (y_predicted_fatal_test.shape)
submission['ConfirmedCases'] = y_predicted_cases_test

submission['Fatalities'] = y_predicted_fatal_test

# submission.head()
submission['ConfirmedCases'] = submission['ConfirmedCases'].apply(np.int64)

submission['Fatalities'] = submission['Fatalities'].apply(np.int64)



print (submission.head())
submission.to_csv('submission.csv', index=False)