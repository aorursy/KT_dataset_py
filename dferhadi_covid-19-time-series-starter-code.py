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



import pandas as pd

import numpy as np

import seaborn as sns

from matplotlib import pyplot as plt

import plotly.graph_objects as go

from fbprophet import Prophet

import pycountry

import plotly.express as px
train = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/train.csv")

test = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/test.csv")

submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-2/submission.csv")

train.head()

train.info()


train.tail()
test.head()
#train = train[train['Date'] < "2020-03-19"]

#train.sample(15)
from sklearn import preprocessing

#train['Lat'] = preprocessing.scale(train['Lat'])

#train['Long'] = preprocessing.scale(train['Long'])

#test['Lat'] = preprocessing.scale(test['Lat'])

#test['Long'] = preprocessing.scale(test['Long'])
# Format date

#train["Date"] = train["Date"].apply(lambda x: x.replace("-",""))

#train["Date"]  = train["Date"].astype(int)

# drop nan's

#train = train.drop(['Province/State'],axis=1)

#train = train.dropna()

train.isnull().sum()
# Do same to Test data

#test["Date"] = test["Date"].apply(lambda x: x.replace("-",""))

#test["Date"]  = test["Date"].astype(int)

# deal with nan's for lat and lon

#test["Lat"]  = test["Lat"].fillna(test['Lat'].mean())

#test["Long"]  = test["Long"].fillna(test['Long'].mean())

test.isnull().sum()
train['Country_Region'].unique()
# Time Series for ConfirmedCases

df = train[train['Country_Region'] == 'Germany']

#df = df[df['Date']]

#df1 = df.drop(['Id','Country/Region','Lat','Long'], axis=1)

confirmed=df.groupby('Date')['ConfirmedCases'].sum().to_frame().reset_index()



#confirmed = df1.drop(['Fatalities'], axis=1)

confirmed['ConfirmedCases'] = np.log(1+confirmed['ConfirmedCases'])

confirmed.plot()

#deaths = df.drop(['ConfirmedCases'], axis=1)


confirmed.columns = ['ds','y']

#confirmed['ds'] = confirmed['ds'].dt.date

confirmed['ds'] = pd.to_datetime(confirmed['ds'])

confirmed.tail()
m = Prophet(interval_width=0.95)

m.fit(confirmed)

future = m.make_future_dataframe(periods=30)

future_confirmed = future.copy() # for non-baseline predictions later on

#future = future[future['ds'].unique()]

future

forecast = m.predict(future)

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
confirmed_forecast_plot = m.plot(forecast)