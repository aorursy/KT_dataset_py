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
train = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_train.csv", parse_dates=['Date'])

train.head()
train.tail()
test = pd.read_csv("/kaggle/input/covid19-local-us-ca-forecasting-week-1/ca_test.csv", parse_dates=['Date'])

test.head()
test.tail()
#Are rows unique?

(len(train)==train.index.nunique()) & (len(test)==test.Date.nunique())
from fbprophet import Prophet

df_c = pd.DataFrame()

df_c['ds'] = pd.to_datetime(train.Date)

df_c['y'] = train.ConfirmedCases

df_c.set_index('ds', inplace=True)

df_c = df_c.loc["2020-03-09":"2020-03-11"]

df_c.reset_index(inplace=True)



df_f = pd.DataFrame()

df_f['ds'] = pd.to_datetime(train.Date)

df_f['y'] = train.Fatalities

df_f.set_index('ds', inplace=True)

df_f = df_f.loc["2020-03-09":"2020-03-11"]

df_f.reset_index(inplace=True)
m_c, m_f = Prophet(), Prophet()

m_c.fit(df_c)

m_f.fit(df_f)



future = m_c.make_future_dataframe(periods=len(test))



forecast_c = m_c.predict(future)

forecast_f = m_f.predict(future)
forecast_c = forecast_c[['ds','yhat']]

forecast_f = forecast_f[['ds','yhat']]



forecast_c= forecast_c.rename(columns={"yhat":"ConfirmedCases"})

forecast_f = forecast_f.rename(columns={"yhat":"Fatalities"})

conf_cases = forecast_c.ConfirmedCases.iloc[3:].reset_index(drop=True)

fatal_cases = forecast_f.Fatalities.iloc[3:].reset_index(drop=True)



test.reset_index(inplace=True,drop=True)
submissions = pd.concat([test.ForecastId, conf_cases,fatal_cases], axis=1)



submissions.head()
submissions.ConfirmedCases = submissions.ConfirmedCases.astype(int)

submissions.Fatalities = submissions.Fatalities.astype(int)
submissions.head()
submissions.to_csv('submission.csv', index=False)