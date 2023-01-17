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
train = pd.read_csv('../input/covid19-global-forecasting-week-3/train.csv')

test = pd.read_csv('../input/covid19-global-forecasting-week-3/test.csv')

submission = pd.read_csv('../input/covid19-global-forecasting-week-3/submission.csv')

clean_train = pd.read_csv('../input/clean-data/clean_train.csv')

clean_test = pd.read_csv('../input/clean-data/clean_test.csv')
tr_p = train[train['Province_State'].notnull()]

tr_p['Country_Province'] = tr_p['Country_Region'] + '_' + tr_p['Province_State']

tr_np = train[train['Province_State'].isnull()]

tr_np['Country_Province'] = tr_np['Country_Region']
train1 = pd.concat([tr_np,tr_p])

train1.drop(['Province_State','Country_Region'], axis=1,inplace = True)
te_p = test[test['Province_State'].notnull()]

te_p['Country_Province'] = te_p['Country_Region'] + '_' + te_p['Province_State']

te_np = test[test['Province_State'].isnull()]

te_np['Country_Province'] = te_np['Country_Region']
test1 = pd.concat([te_np,te_p])

test1.drop(['Province_State','Country_Region'], axis=1,inplace = True)
train1['Date'] = pd.to_datetime(train1['Date'])

test1['Date'] = pd.to_datetime(test1['Date'])
train1['Country_Province'] = train1['Country_Province'].astype('category')

test1['Country_Province'] = test1['Country_Province'].astype('category')
cl_tr1 = clean_train[['Country_Region','Province_State','Lat','Long','firstcase','density','medianage','urbanpop','hospibed','lung','avgtemp','avghumidity']]
cl_tr1.drop_duplicates(subset=None, keep='first', inplace=True)
cl_p = cl_tr1[cl_tr1['Province_State'].notnull()]

cl_p['Country_Province'] = cl_p['Country_Region'] + '_' + cl_p['Province_State']

cl_np = cl_tr1[cl_tr1['Province_State'].isnull()]

cl_np['Country_Province'] = cl_np['Country_Region']
cl_tr = pd.concat([cl_p,cl_np])

cl_tr.drop(['Country_Region','Province_State'], axis = 1, inplace = True)
train_cl = pd.merge(train1,cl_tr)

test_cl = pd.merge(test1,cl_tr)
train_cl['Country_Province'] = train_cl['Country_Province'].astype('category')

test_cl['Country_Province'] = test_cl['Country_Province'].astype('category')
train_cl['firstcase'] = pd.to_datetime(train_cl['firstcase'])

test_cl['firstcase'] = pd.to_datetime(test_cl['firstcase'])
from datetime import datetime

train_cl['jan']="2020-01-01"

train_cl['jan'] = pd.to_datetime(train_cl['jan'])

test_cl['jan']="2020-01-01"

test_cl['jan'] = pd.to_datetime(test_cl['jan'])



train_cl['days_since_jan1'] = train_cl['Date']-train_cl['jan']

test_cl['days_since_jan1'] = test_cl['Date']-test_cl['jan']
for i in range(len(train_cl)):

               train_cl['days_since_jan1'][i]=train_cl['days_since_jan1'][i].days

        

for i in range(len(test_cl)):

               test_cl['days_since_jan1'][i]=test_cl['days_since_jan1'][i].days
train_cl['days_since_firstcase'] = train_cl['firstcase']-train_cl['Date']

test_cl['days_since_firstcase'] = test_cl['firstcase']-test_cl['Date']
for i in range(len(train_cl)):

               train_cl['days_since_firstcase'][i]=train_cl['days_since_firstcase'][i].days

        

for i in range(len(test_cl)):

               test_cl['days_since_firstcase'][i]=test_cl['days_since_firstcase'][i].days
cols = ['days_since_jan1','days_since_firstcase']



for col in cols:

    train_cl[col] = train_cl[col].astype('int64')

    test_cl[col] = test_cl[col].astype('int64')
train_clean_cases = train_cl[['Lat', 'Long','density', 'medianage', 'urbanpop',

                        'hospibed','lung', 'avgtemp', 'avghumidity','days_since_jan1', 'days_since_firstcase']]



test_clean_cases = test_cl[['Lat', 'Long','density', 'medianage', 'urbanpop',

                        'hospibed','lung', 'avgtemp', 'avghumidity','days_since_jan1', 'days_since_firstcase']]
train_clean_fatal = train_cl[['Lat', 'Long','density', 'medianage', 'urbanpop',

                        'hospibed','lung', 'avgtemp', 'avghumidity','days_since_jan1', 'days_since_firstcase','ConfirmedCases']]



test_clean_fatal = test_cl[['Lat', 'Long','density', 'medianage', 'urbanpop',

                        'hospibed','lung', 'avgtemp', 'avghumidity','days_since_jan1', 'days_since_firstcase']]
train_y1 = train_cl['ConfirmedCases']

train_y2 = train_cl['Fatalities']
from sklearn.tree import DecisionTreeRegressor

dt_1=DecisionTreeRegressor(max_depth=30,max_features=8,min_samples_split=2,min_samples_leaf=1)

dt_2=DecisionTreeRegressor(max_depth=30,max_features=8,min_samples_split=2,min_samples_leaf=1)

dt_1.fit(train_clean_cases,train_y1)

dt_2.fit(train_clean_fatal,train_y2)
dt_train_cases_pred = dt_1.predict(train_clean_cases)

dt_train_fatal_pred = dt_2.predict(train_clean_fatal)
from sklearn.metrics import mean_squared_error

dt_mse_train_cases = mean_squared_error(dt_train_cases_pred, train_y1)

dt_rmse_train_cases = np.sqrt(dt_mse_train_cases)

print("DT Regression MSE on train cases: %.4f" %dt_mse_train_cases)

print('DT Regression RMSE on train cases: %.4f' % dt_rmse_train_cases)
dt_mse_train_fatalities = mean_squared_error(dt_train_fatal_pred, train_y2)

dt_rmse_train_fatalities = np.sqrt(dt_mse_train_fatalities)

print("DT Regression MSE on train fatalities: %.4f" %dt_mse_train_cases)

print('DT Regression RMSE on train fatalities: %.4f' % dt_rmse_train_cases)
dt_test_cases_pred = dt_1.predict(test_clean_cases)

dt_test_cases_pred = np.where(dt_test_cases_pred<0,0,np.rint(dt_test_cases_pred))
test_clean_fatal['ConfirmedCases']= dt_test_cases_pred
dt_test_fatal_pred = dt_2.predict(test_clean_fatal)
submission['ForecastId'] = test_cl['ForecastId']

submission['ConfirmedCases'] = dt_test_cases_pred

submission['Fatalities'] = dt_test_fatal_pred
submission.to_csv('submission.csv',index=False)