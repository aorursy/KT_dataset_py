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
from fbprophet import Prophet 
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/train.csv', parse_dates=True)
test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/test.csv', parse_dates=True)
submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-3/submission.csv')
train.tail()
test.head()
test.tail()
train['ds'] = pd.to_datetime(train.Date)
test['ds'] = pd.to_datetime(test.Date)
train.columns
train.columns = ['Id', 'Province_State', 'Country_Region', 'Date', 'y', 'Fatalities', 'ds']
train.info()
test.info()
train['Province_State'].fillna('0', inplace=True)
test['Province_State'].fillna('0', inplace=True)
train['unique'] = train[['Province_State', 'Country_Region']].agg('-'.join, axis=1)
test['unique'] = test[['Province_State', 'Country_Region']].agg('-'.join, axis=1)
test['predicted_cases'] = 0
test['predicted_fatalitiies'] = 0
test.tail()
indexes = train['unique'].unique()
i = 0

for name in indexes:

  group = train.loc[train['unique']==name, ['ds', 'y']]

  p = Prophet(n_changepoints=10, changepoint_prior_scale=100, changepoint_range=0.99)
  p.fit(group)

  df = test.loc[test['unique']==name, ['ds']]
  forecast = p.predict(df)[['ds', 'yhat']]
  
  test.loc[(test['unique']==name) & (test['ds']<='2020-04-04'), 'predicted_cases'] = np.array(group.loc[(group['ds']>='2020-03-26') & (group['ds']<='2020-04-04'), 'y'])
  test.loc[(test['unique']==name) & (test['ds']>'2020-04-04'), 'predicted_cases'] = np.array(forecast.loc[forecast['ds']>'2020-04-04', 'yhat'])

  print(i, " ", name, "done")
  i = i+1

#((test['unique']=='0-Afghanistan') & (test['ds']>'2020-04-04')).sum()
#test.loc[(test['unique']=='0-Afghanistan') & (test['ds']>'2020-04-04'), 'predicted_cases']
#len(np.array(forecast.loc[forecast['ds']>'2020-04-04', 'yhat']))
#np.array(group.loc[group['ds']>='2020-03-26', ['ds', 'y']])
#len(np.array(group.loc[(group['ds']>='2020-03-26') & (group['ds']<='2020-04-04'), 'y']))
#test.iloc[:60,:]
train.columns
train.columns =  ['Id', 'Province_State', 'Country_Region', 'Date', 'ConfirmedCases', 'y', 'ds', 'unique']
indexes = train['unique'].unique()
i = 0

for name in indexes:

  group = train.loc[train['unique']==name, ['ds', 'y']]

  p = Prophet(n_changepoints=10, changepoint_prior_scale=100, changepoint_range=0.99)
  p.fit(group)

  df = test.loc[test['unique']==name, ['ds']]
  forecast = p.predict(df)[['ds', 'yhat']]
  
  #test.loc[test['unique']==name, 'predicted_fatalitiies'] = np.array(forecast['yhat'])
  test.loc[(test['unique']==name) & (test['ds']<='2020-04-04'), 'predicted_fatalitiies'] = np.array(group.loc[(group['ds']>='2020-03-26') & (group['ds']<='2020-04-04'), 'y'])
  test.loc[(test['unique']==name) & (test['ds']>'2020-04-04'), 'predicted_fatalitiies'] = np.array(forecast.loc[forecast['ds']>'2020-04-04', 'yhat'])
    
  print(i, " ", name, "done")
  i = i+1
submission.loc[:,'ConfirmedCases'] = test['predicted_cases']
submission.loc[:,'Fatalities'] = test['predicted_fatalitiies']
submission.to_csv('submission.csv', index = False)
