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
import pandas as pd

from pathlib import Path

from pandas_profiling import ProfileReport

from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import LabelEncoder

import datetime

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

from sklearn.model_selection import cross_val_score
dataset_path = Path('/kaggle/input/covid19-global-forecasting-week-4')



train = pd.read_csv(dataset_path/'train.csv')

test = pd.read_csv(dataset_path/'test.csv')

submission = pd.read_csv(dataset_path/'submission.csv')
def fill_state(state,country):

    if pd.isna(state) : return country

    return state
train['Province_State'] = train.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fill_state(x['Province_State'], x['Country_Region']), axis=1)

test['Province_State'] = test.loc[:, ['Province_State', 'Country_Region']].apply(lambda x : fill_state(x['Province_State'], x['Country_Region']), axis=1)
train['Date'] = pd.to_datetime(train['Date'],infer_datetime_format=True)

test['Date'] = pd.to_datetime(test['Date'],infer_datetime_format=True)



train['Day_of_Week'] = train['Date'].dt.dayofweek

test['Day_of_Week'] = test['Date'].dt.dayofweek



train['Month'] = train['Date'].dt.month

test['Month'] = test['Date'].dt.month



train['Day'] = train['Date'].dt.day

test['Day'] = test['Date'].dt.day



train['Day_of_Year'] = train['Date'].dt.dayofyear

test['Day_of_Year'] = test['Date'].dt.dayofyear



train['Week_of_Year'] = train['Date'].dt.weekofyear

test['Week_of_Year'] = test['Date'].dt.weekofyear



train['Quarter'] = train['Date'].dt.quarter  

test['Quarter'] = test['Date'].dt.quarter  



train.drop('Date',1,inplace=True)

test.drop('Date',1,inplace=True)
submission=pd.DataFrame(columns=submission.columns)



l1=LabelEncoder()

l2=LabelEncoder()



l1.fit(train['Country_Region'])

l2.fit(train['Province_State'])
countries=train['Country_Region'].unique()

for country in countries:

    country_df=train[train['Country_Region']==country]

    provinces=country_df['Province_State'].unique()

    for province in provinces:

            train_df=country_df[country_df['Province_State']==province]

            train_df.pop('Id')

            x=train_df[['Province_State','Country_Region','Day_of_Week','Month','Day','Day_of_Year','Week_of_Year','Quarter']]

            x['Country_Region']=l1.transform(x['Country_Region'])

            x['Province_State']=l2.transform(x['Province_State'])

            y1=train_df[['ConfirmedCases']]

            y2=train_df[['Fatalities']]

            model_1=DecisionTreeClassifier()

            model_2=DecisionTreeClassifier()

            model_1.fit(x,y1)

            model_2.fit(x,y2)

            test_df=test.query('Province_State==@province & Country_Region==@country')

            test_id=test_df['ForecastId'].values.tolist()

            test_df.pop('ForecastId')

            test_x=test_df[['Province_State','Country_Region','Day_of_Week','Month','Day','Day_of_Year','Week_of_Year','Quarter']]

            test_x['Country_Region']=l1.transform(test_x['Country_Region'])

            test_x['Province_State']=l2.transform(test_x['Province_State'])

            test_y1=model_1.predict(test_x)

            test_y2=model_2.predict(test_x)

            test_res=pd.DataFrame(columns=submission.columns)

            test_res['ForecastId']=test_id

            test_res['ConfirmedCases']=test_y1

            test_res['Fatalities']=test_y2

            submission=submission.append(test_res)
submission

submission.to_csv('submission.csv',index=False)