# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime as dt

from sklearn import preprocessing

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import SGDRegressor

import lightgbm as lgb

import statistics as s

from sklearn.model_selection import cross_validate

from tqdm import tqdm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")
display(train.head())

display(test.head())
def date_processor(string):

    string = dt.strptime(string, '%Y-%m-%d').date()

#     print(type(string.toordinal()))

    return string.toordinal()
train['Date'] = train['Date'].apply(date_processor)

# train.drop(['Province_State'], axis=1, inplace = True)

test['Date'] = test['Date'].apply(date_processor)

# test.drop(['Province_State'], axis=1, inplace = True)
# train.set_index('Id', inplace=True)

# test.set_index('ForecastId', inplace=True)
train = train.rename(columns={

    'Country_Region' : 'cr',

    'Province_State' : 'ps',

    'Fatalities' : 'dead',

    'ConfirmedCases' : 'cases',

    'Date' : 'date',

})

test = test.rename(columns={

    'Country_Region' : 'cr',

    'Province_State' : 'ps',

    'Date' : 'date',

})
# train.head()

print(type(train.ps.iloc[0]) == float)

# train.ps.iloc[0] == None
print(set(train.cr) == set(test.cr))

print(set(train.ps) == set(test.ps))
def fill_state(state ,country):

    if type(state) == float:

        return country

    else:

        return state
train.ps = train.loc[:,['ps','cr']].apply(lambda x: fill_state(x['ps'],x['cr']), axis = 1)

test.ps = test.loc[:,['ps','cr']].apply(lambda x: fill_state(x['ps'],x['cr']), axis = 1)

display(train.head())

display(test.head())
le = preprocessing.LabelEncoder()

train.cr = le.fit_transform(train.cr)

test.cr = le.fit_transform(test.cr)

le_state = preprocessing.LabelEncoder()

train.ps = le.fit_transform(train.ps)

test.ps = le.fit_transform(test.ps)
# train.date = train.date / max(train.date)
pd.get_dummies(train, columns = train.columns[1:4])
# train = pd.get_dummies(train, columns = train.columns[1:4])
# test = pd.get_dummies(test, columns = test.columns[1:2])
print(test.columns)

print(submission.shape)
from sklearn.tree import DecisionTreeRegressor
countries = list(train.cr.unique())



df_out = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

df_out2 = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})



score_mean_t1 = []

score_mean_t2 = []



CV = 5

N_ESTIMATORS = 500



sgd = DecisionTreeRegressor()

rf = RandomForestRegressor(n_estimators = N_ESTIMATORS)

models = [sgd]

for model in models:

    for c in tqdm(countries,position=0, leave=True):

        states = train.loc[train.cr == c, :].ps.unique()

        for s in states:

            train_data = train.loc[(train.ps==s) & (train.cr==c),['ps','cr','date']] 

            train_data = pd.get_dummies(train_data, columns = train_data.columns[1:4])

            t1 = train.loc[(train.ps==s) & (train.cr==c),'cases']

            t2 = train.loc[(train.ps==s) & (train.cr==c),'dead']



            test_data  = test.loc[(test.ps==s) & (test.cr==c),['ps','cr','date']]

            ids = test.loc[(test.ps==s) & (test.cr==c),'ForecastId']

            test_data = pd.get_dummies(test_data,  columns = test.columns[1:2])

            model1 = model

#             cv_result = cross_validate(model1, train_data, t1, scoring = 'r2', cv= CV)

#             print("CV Test Score : Mean : %.7g" % (np.mean(cv_result['test_score'])))

#             score_mean_t1.append(np.mean(cv_result['test_score']))



            model1.fit(train_data, t1)

            t1_pred = model1.predict(test_data)



            model2 = model

#             cv_result = cross_validate(model2, train_data, t2,scoring = 'r2', cv= CV)

#             print("CV Test Score : Mean : %.7g" % (np.mean(cv_result['test_score'])))

#             score_mean_t2.append(np.mean(cv_result['test_score']))

            model2.fit(train_data,t2)

            t2_pred = model2.predict(test_data)



            # LightGBM

    #         model3 = lgb.LGBMRegressor(n_estimators=2000)

    #         model3.fit(train_data, t1)

    #         t3_pred = model3.predict(test_data)



    #         model4 = lgb.LGBMRegressor(n_estimators=2000)

    #         model4.fit(train_data, t2)

    #         t4_pred = model4.predict(test_data)



            df = pd.DataFrame({'ForecastId': ids, 'ConfirmedCases': t1_pred, 'Fatalities': t2_pred})

    #         df2 = pd.DataFrame({'ForecastId': ids, 'ConfirmedCases': t3_pred, 'Fatalities': t4_pred})

            df_out = pd.concat([df_out, df], axis=0)

    #         df_out2 = pd.concat([df_out2, df2], axis=0)

    print("T1 mean test score for : ",model, "is :" , np.mean(score_mean_t1))

    print("T1 mean test score for : ",model, "is :" , np.mean(score_mean_t2))
countries[161:]
df_out.ForecastId = df_out.ForecastId.astype('int')

# df_out2.ForecastId = df_out2.ForecastId.astype('int')
# df_out['ConfirmedCases'] = (1/2)*(df_out['ConfirmedCases'] + df_out2['ConfirmedCases'])

# df_out['Fatalities'] = (1/2)*(df_out['Fatalities'] + df_out2['Fatalities'])
df_out['ConfirmedCases'] = df_out['ConfirmedCases'].round().astype(int)

df_out['Fatalities'] = df_out['Fatalities'].round().astype(int)
print(df_out.shape)

df_out.to_csv('submission.csv', index=False)