# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
sub = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

train['Date'] = pd.to_datetime(train['Date'])
test['Date'] = pd.to_datetime(test['Date'])


train['dayofmonth'] = train['Date'].dt.day
train['dayofweek'] = train['Date'].dt.dayofweek
train['month'] = train['Date'].dt.month
train['weekNumber'] = test['Date'].dt.week
train['dayofyear'] = train['Date'].dt.dayofyear
train['Fatalities_ratio'] = train['Fatalities'] / train['ConfirmedCases']

train['Change_ConfirmedCases'] = train.groupby(np.where(train['Province_State'].isnull(), train['Country_Region'], train['Province_State'])).ConfirmedCases.pct_change()
train['Change_Fatalities'] = train.groupby(np.where(train['Province_State'].isnull(), train['Country_Region'], train['Province_State'])).Fatalities.pct_change()


test['dayofmonth'] = test['Date'].dt.day
test['dayofweek'] = test['Date'].dt.dayofweek
test['month'] = test['Date'].dt.month
test['weekNumber'] = test['Date'].dt.week
test['dayofyear'] = test['Date'].dt.dayofyear


train["Country_Region_"] = train[["Country_Region", "Province_State"]].apply(lambda x : x[0]+'_'+x[1] if type(x[1]) == str else x[0] , axis=1)
test["Country_Region_"] = test[["Country_Region", "Province_State"]].apply(lambda x : x[0]+'_'+x[1] if type(x[1]) == str else x[0], axis=1)


train = train.set_index('Date')
test = test.set_index('Date')
train_c = train['Country_Region_'].values
test_c = test['Country_Region_'].values
for x in ['Province_State','Country_Region','Country_Region_']:
    train[x] = le.fit_transform(train[x].fillna('0'))
    test[x] = le.fit_transform(test[x].fillna('0'))

x_cols = ['Country_Region_','dayofmonth','dayofweek','month','weekNumber','dayofyear','Province_State','Country_Region']
x_cols_fatal = x_cols + ['ConfirmedCases']
change_pct_cols = ['Change_ConfirmedCases','Change_Fatalities','Fatalities_ratio']
train= train.fillna(0)
test= test.fillna(0)
train
targets = ['ConfirmedCases', 'Fatalities']
%%time
xgbr_conf = xgb.XGBRegressor(n_estimators=1000,objective = "count:poisson")
xgbr_conf.fit(train[x_cols], train[[targets[0]]],verbose=False)

xgbr_conf_change = xgb.XGBRegressor(n_estimators=1000,objective = "count:poisson")
xgbr_conf_change.fit(train[x_cols + change_pct_cols], train[[targets[0]]],verbose=False)
%%time
xgbr_fatal = xgb.XGBRegressor(n_estimators=1000,objective = "count:poisson")
xgbr_fatal.fit(train[x_cols_fatal], train[[targets[1]]],verbose=False)
t = test[x_cols].copy()
t['ConfirmedCases'] =  xgbr_conf.predict(test[x_cols]).astype(int)
t['Fatalities'] = xgbr_fatal.predict(t).astype(int)
t['Fatalities_ratio'] = t['Fatalities'] / t['ConfirmedCases']

t['Change_ConfirmedCases'] = t.groupby(np.where(t['Province_State'].isnull(), t['Country_Region'], t['Province_State'])).ConfirmedCases.pct_change()
t['Change_Fatalities'] = t.groupby(np.where(t['Province_State'].isnull(), t['Country_Region'], t['Province_State'])).Fatalities.pct_change()
t
t = t.fillna(0)
%%time
xgbr_fatal_change = xgb.XGBRegressor(n_estimators=1000,objective = "count:poisson")
xgbr_fatal_change.fit(train[x_cols_fatal + change_pct_cols], train[[targets[1]]],verbose=False)
p = t[x_cols + change_pct_cols].copy()
p
p['ConfirmedCases'] =  xgbr_conf_change.predict(p[x_cols + change_pct_cols]).astype(int)
p
p['Fatalities'] = xgbr_fatal_change.predict(p[x_cols_fatal + change_pct_cols]).astype(int)
p
p
p[p.c_region == 'Romania'][['ConfirmedCases','Fatalities']].plot(figsize=(20,10))
%%time
sub.ConfirmedCases = p.ConfirmedCases.values
sub.Fatalities = p.Fatalities.values
sub.to_csv('submission.csv',index=False)