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
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
train_original = train.copy()

test_original = test.copy()
def fill_province(row):

    province = row.Province_State

    country = row.Country_Region

    if pd.isnull(province):

        return country

    else:

        return province
train['Province_State'] = train.apply(fill_province, axis=1)

test['Province_State']= test.apply(fill_province, axis=1)
from sklearn.preprocessing import LabelEncoder
label_encoder1 = LabelEncoder()

label_encoder2 = LabelEncoder()
train['Province_State_encoded'] = label_encoder1.fit_transform(train['Province_State'])

test['Province_State_encoded'] = label_encoder1.transform(test['Province_State'])
train['Country_Region_encoded'] = label_encoder2.fit_transform(train['Country_Region'])

test['Country_Region_encoded'] = label_encoder2.transform(test['Country_Region'])
train.head()
train['Date'] = pd.to_datetime(train_original.Date, infer_datetime_format=True)

train['Date'] = train.Date.dt.strftime('%y%m%d')

train['Date'] = train['Date'].astype(int)

test['Date'] = pd.to_datetime(test_original.Date, infer_datetime_format=True)

test['Date'] = test.Date.dt.strftime('%y%m%d')

test['Date'] = test['Date'].astype(int)
train.head()
test.head()
train.tail()
test.tail()
from xgboost import XGBRegressor
X = train[['Province_State_encoded', 'Country_Region_encoded', 'Date']]

y = train[['ConfirmedCases', 'Fatalities']]
y_confirmed = y['ConfirmedCases']

y_fatalities = y['Fatalities']
model1 = XGBRegressor(n_estimators=40000)
model1.fit(X,y_confirmed)
X_test = test[['Province_State_encoded', 'Country_Region_encoded', 'Date']]
y_confirmed_pred = model1.predict(X_test)
model2 = XGBRegressor(n_estimators=30000)
model2.fit(X,y_fatalities)
y_fatalities_pred = model2.predict(X_test)
submission_dict = {'ForecastId': test['ForecastId'], 'ConfirmedCases':y_confirmed_pred, 'Fatalities':y_fatalities_pred}
submission_df = pd.DataFrame(submission_dict)
submission_df.head()
submission_df = submission_df.clip(0)
submission_df.to_csv('submission.csv', index=False)