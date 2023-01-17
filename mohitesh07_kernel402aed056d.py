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
from sklearn.tree import DecisionTreeRegressor

from sklearn import preprocessing
df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')

df.head()
df['Date'] = pd.to_datetime(df['Date'])

df['Date'] = pd.to_timedelta(df.Date).dt.total_seconds().astype(int)



df = df.replace(np.nan, '', regex=True)



le_cr = preprocessing.LabelEncoder()

le_ps = preprocessing.LabelEncoder()



le_cr.fit(df['Country_Region'])

le_ps.fit(df['Province_State'])



df['Country_Region'] = le_cr.transform(df['Country_Region'])

df['Province_State'] = le_ps.transform(df['Province_State'])
df.head()
regressor_cc = DecisionTreeRegressor(random_state = 0, max_features="sqrt") 

regressor_f = DecisionTreeRegressor(random_state = 0, max_features="sqrt") 



regressor_cc.fit(df[['Country_Region','Province_State','Date']], df[['ConfirmedCases']]) 

regressor_f.fit(df[['Country_Region','Province_State','Date']], df[['Fatalities']]) 
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')
df_test['Date'] = pd.to_datetime(df_test['Date'])

df_test['Date'] = pd.to_timedelta(df_test.Date).dt.total_seconds().astype(int)



df_test = df_test.replace(np.nan, '', regex=True)



df_test['Country_Region'] = le_cr.transform(df_test['Country_Region'])

df_test['Province_State'] = le_ps.transform(df_test['Province_State'])
df_s = pd.DataFrame()



df_s['ForecastId'] = df_test['ForecastId']

df_s['ConfirmedCases'] = regressor_cc.predict(df_test[['Country_Region','Province_State','Date']])

df_s['Fatalities'] = regressor_f.predict(df_test[['Country_Region','Province_State','Date']])
df_s
df_s.to_csv('submission.csv', index=False)