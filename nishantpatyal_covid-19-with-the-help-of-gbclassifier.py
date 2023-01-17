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
import numpy as np

import pandas as pd

import category_encoders as ce

from sklearn.ensemble import GradientBoostingRegressor
train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv', index_col = 'Id') 

test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
train['Province_State'].fillna('Nan',inplace = True)

test['Province_State'].fillna('Nan',inplace = True)
OE = ce.OrdinalEncoder()

train['Province_State'] = OE.fit_transform(train['Province_State'])
OE = ce.OrdinalEncoder()

train['Country_Region'] = OE.fit_transform(train['Country_Region'])
OE = ce.OrdinalEncoder()

test['Country_Region'] = OE.fit_transform(test['Country_Region'])
OE = ce.OrdinalEncoder()

test['Province_State'] = OE.fit_transform(test['Province_State'])
train['Date'] = pd.to_datetime(train['Date'])

train['Month'] = train['Date'].dt.month

train['Day'] = train['Date'].dt.day
test['Date'] = pd.to_datetime(test['Date'])

test['Month'] =  test['Date'].dt.month

test['Day'] =  test['Date'].dt.day
C = ['Province_State', 'Country_Region','Month', 'Day']
train = train[train['Date']<test.Date.min()]
train.drop('Date',1,inplace = True)

test.drop('Date',1,inplace = True)
df_train = train

df_test  =test
df_train
submission = []

for country in df_train.Country_Region.unique():

    df_train1 = df_train[df_train["Country_Region"]==country]

   

    for state in df_train1.Province_State.unique():

        df_train2 = df_train1[df_train1["Province_State"]==state]

       

        train = df_train2.values

        X_train, y_train = train[:,:-2], train[:,-2:]

        

        

        model1 = GradientBoostingRegressor(n_estimators=100)

        model1.fit(X_train, y_train[:,0])

        

        

        model2 = GradientBoostingRegressor(n_estimators=200)

        model2.fit(X_train, y_train[:,1])

        

        df_test1 = df_test[(df_test["Country_Region"]==country) & (df_test["Province_State"] == state)]

        

        ForecastId = df_test1.ForecastId.values

        

        df_test2 = df_test1[C]

        

        y_pred1 = model1.predict(df_test2.values)

        y_pred2 = model2.predict(df_test2.values)

        

        for i in range(len(y_pred1)):

            d = {'ForecastId':ForecastId[i], 'ConfirmedCases':y_pred1[i], 'Fatalities':y_pred2[i]}

            submission.append(d)
pd.DataFrame(submission).to_csv(r'submission.csv', index=False)