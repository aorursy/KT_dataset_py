# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from warnings import filterwarnings

filterwarnings('ignore')

from sklearn import preprocessing

from xgboost import XGBRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

submission = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/submission.csv')
train.head()
train.describe()
train.info()
test.head()
test.describe()
submission.head()
train.rename(columns={'Country_Region':'Countries'}, inplace=True)

test.rename(columns={'Country_Region':'Countries'}, inplace=True)



train.rename(columns={'Province_State':'States'}, inplace=True)

test.rename(columns={'Province_State':'States'}, inplace=True)



train['Date'] = pd.to_datetime(train['Date'], infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'], infer_datetime_format=True)
train.info()

test.info()
y1T = train.iloc[:, -2]

y1T.head()
y2T = train.iloc[:, -1]

y2T.head()
na = "na"

def filtrar_state(states, countries):    

    if states == na:

        return countries

    else: 

        return states
train_1 = train.copy()

train_1['States'].fillna(na, inplace=True)

train_1['States'] = train_1.loc[:, ['States', 'Countries']].apply(lambda x : filtrar_state(x['States'], x['Countries']), axis=1)

train_1.loc[:, 'Date'] = train_1.Date.dt.strftime("%m%d")

train_1["Date"]  = train_1["Date"].astype(int)

train_1.head()
te = test.copy()

te['States'].fillna(na, inplace=True)

te['States'] = te.loc[:, ['States', 'Countries']].apply(lambda x : filtrar_state(x['States'], x['Countries']), axis=1)

te.loc[:, 'Date'] = te.Date.dt.strftime("%m%d")

te["Date"]  = te["Date"].astype(int)

te.head()
le = preprocessing.LabelEncoder()



train_1.Countries = le.fit_transform(train_1.Countries)

train_1['States'] = le.fit_transform(train_1['States'])

train_1.head()
te.Countries = le.fit_transform(te.Countries)

te['States'] = le.fit_transform(te['States'])



te.head()
train.head()
train.loc[train.Countries == 'Afghanistan', :]
test.tail()
output = pd.DataFrame({'ForecastId': [], 'ConfirmedCases': [], 'Fatalities': []})

le = preprocessing.LabelEncoder()

countr = train_1.Countries.unique()

for countries in countr:

    states = train_1.loc[train_1.Countries == countries, :].States.unique()

    for state in states:

        X_tra = train_1.loc[(train_1.Countries == countries) & (train_1.States == state), ['State', 'Countries', 'Date',

                                                                                                    'ConfirmedCases','Fatalities']]

        y_cc_train = X_tra.loc[:, 'ConfirmedCases']

        y_fat_train = X_tra.loc[:, 'Fatalities']

        

        X_tra = X_tra.loc[:, ['States', 'Countries', 'Date']]

        

        X_tra.Countries = le.fit_transform(X_tra.Countries)

        X_tra['States'] = le.fit_transform(X_tra['States'])

        

        X_te = te.loc[(te.Countries == countries) & (te.States == state), ['States', 'Countries', 'Date', 

                                                                                                'ForecastId']]        

        X_te_Id = X_te.loc[:, 'ForecastId']

        X_te = X_te.loc[:, ['States', 'Countries', 'Date']]

        

        X_te.Countries = le.fit_transform(X_te.Countries)

        X_te['States'] = le.fit_transform(X_te['States'])

        

        #Confirmed Cases

        xmodel1 = XGBRegressor(learning_rate =0.01,n_estimators=2000,objective='count:poisson',max_depth=6,min_child_weight=4,gamma=0,

                               subsample=0.8,colsample_bytree=0.8,nthread=4,scale_pos_weight=1)

        #Fit_Confirmed_cases

        xmodel1.fit(X_tra, y_cc_train, eval_metric='auc')

        #Predict_Confirmed_cases

        y_cc_pred = xmodel1.predict(X_te)

        

        

        #Fatalities

        xmodel2 = XGBRegressor(learning_rate =0.01,n_estimators=2000,objective = 'count:poisson',max_depth=6,min_child_weight=4,gamma=0,

            subsample=0.8,colsample_bytree=0.8,nthread=4,scale_pos_weight=1)

        #Fit_Fatalities

        xmodel2.fit(X_tra, y_fat_train, eval_metric='auc')

        #Predict_Fatalities

        y_fat_pred = xmodel2.predict(X_te)

        

        dt = pd.DataFrame({'ForecastId': X_te_Id, 'ConfirmedCases': y_cc_pred, 'Fatalities': y_fat_pred})

        output = pd.concat([output, dt], axis=0)
output.ForecastId = output.ForecastId.astype('int')

output.tail()
output.to_csv('submission.csv', index=False)
output.head()