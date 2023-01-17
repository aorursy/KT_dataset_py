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
# ML libraries
import lightgbm as lgb
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import BayesianRidge 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_log_error
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")
test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")
train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")
display(train.head())
display(test.head())
display(train.describe())
print("No of colums and name in training datasets are")
print(train.columns.values)
print("No of Country regions")
print(np.unique(train['Country_Region'].values))
print("No of countries to have country/Provinces provided")
print(train[train['Province_State'].isna()==False]['Country_Region'].unique())
#count no of null in columns
print("id",train['Id'].isnull().sum())
print("Province_State",train['Province_State'].isnull().sum())
print("Coutry_Region",train['Country_Region'].isnull().sum())
print("Date",train['Date'].isnull().sum())
print("ConfirmedCases",train['ConfirmedCases'].isnull().sum())
print("Fatalities",train['Fatalities'].isnull().sum())
print("For Training dataset:-")
print("Starting date for corona cases",min(train['Date']))
print("SO far corona cases date",max(train['Date']))
print("Total no of days",train['Date'].nunique())
print("For Test Dataset:-")
print("Starting date for corona cases",min(test['Date']))
print("SO far corona cases date",max(test['Date']))
print("Total no of days",test['Date'].nunique())

#Adding features related to date
train['Date']= pd.to_datetime(train['Date'])
test['Date']=pd.to_datetime(test['Date'])

train['dayofmonth']=train['Date'].dt.day
train['dayofweek'] = train['Date'].dt.dayofweek
train['month'] = train['Date'].dt.month
train['weekNumber'] = train['Date'].dt.week
train['dayofyear'] = train['Date'].dt.dayofyear


test['dayofmonth']=test['Date'].dt.day
test['dayofweek'] = test['Date'].dt.dayofweek
test['month'] = test['Date'].dt.month
test['weekNumber'] = test['Date'].dt.week
test['dayofyear'] = test['Date'].dt.dayofyear

## added in training set
train['Fatalities_ratio'] = train['Fatalities'] / train['ConfirmedCases']

train['Fatalities_ratio'].fillna(0,inplace=True)

# Replacing all the Province_State that are null by the Country_Region values
train.Province_State.fillna(train['Country_Region'], inplace=True)
test.Province_State.fillna(test['Country_Region'], inplace=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

train.Country_Region = le.fit_transform(train.Country_Region)
train['Province_State'] = le.fit_transform(train['Province_State'])

test.Country_Region = le.fit_transform(test.Country_Region)
test['Province_State'] = le.fit_transform(test['Province_State'])


y1_train = train['ConfirmedCases']
y2_train = train['Fatalities']
X_Id = train['Id']

X_train = train.drop(columns=['Id', 'Date','ConfirmedCases', 'Fatalities','Fatalities_ratio'])
X_test  = test.drop(columns=['ForecastId', 'Date'])

print(X_train.shape)
print(X_test.shape)
print(X_train.columns.values)
print(X_test.columns.values)
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y1_train)
y1_pred_rf = rf.predict(X_test)

rf.fit(X_train, y2_train)
y2_pred_rf = rf.predict(X_test)

rf_pred = pd.DataFrame({'ForecastId': test.ForecastId, 'ConfirmedCases': y1_pred_rf, 'Fatalities': y2_pred_rf})
rf_pred.to_csv('submission.csv', index=False)

