#Change the country name and go all the way down to see the predictions for today

COUNTRY = "Sri Lanka"
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

import matplotlib.pyplot as plt 

import matplotlib.colors as mcolors

import pandas as pd 

import random

import math

import time

from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime

import operator 

from datetime import datetime

plt.style.use('seaborn')

%matplotlib inline 
confirmed_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

death_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

recovered_df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

confirmed_df.head()
confirmed_df_t = confirmed_df.melt(

    id_vars = ["Country/Region", "Province/State","Lat","Long"],

    var_name = "Date",

    value_name="Value"

)

confirmed_df_t.head()
if(COUNTRY=="World"):

    confirmed_df_t_sl = confirmed_df_t.drop(["Country/Region","Province/State","Lat","Long"], axis=1)

    confirmed_df_t_sl = confirmed_df_t_sl.groupby(["Date"]).sum()





else:

    confirmed_df_t_sl = confirmed_df_t[confirmed_df_t['Country/Region']==COUNTRY]

    confirmed_df_t_sl = confirmed_df_t_sl.drop(["Province/State","Lat","Long"], axis=1)

    confirmed_df_t_sl = confirmed_df_t_sl.groupby(["Country/Region","Date"]).sum()



confirmed_df_t_sl = confirmed_df_t_sl.reset_index()

confirmed_df_t_sl["Date"] = pd.to_datetime(confirmed_df_t_sl['Date'])

confirmed_df_t_sl = confirmed_df_t_sl.sort_values(by=["Date"])

confirmed_df_t_sl.tail()
confirmed_df_t_sl['frequency'] = confirmed_df_t_sl[['Value']].diff().fillna(confirmed_df_t_sl)

confirmed_df_t_sl.tail()
# from IPython.display import HTML

# confirmed_df_t.to_csv('submission.csv')



# def create_download_link(title = "Download CSV file", filename = "data.csv"):  

#     html = '<a href={filename}>{title}</a>'

#     html = html.format(title=title,filename=filename)

#     return HTML(html)



# # create a link to download the dataframe which was saved with .to_csv method

# create_download_link(filename='submission.csv')
confirmed_df_t_sl["Date"] = pd.to_datetime(confirmed_df_t_sl['Date'])

confirmed_df_t_sl_ts = confirmed_df_t_sl.iloc[:,-3:]

confirmed_df_t_sl_ts = confirmed_df_t_sl_ts.set_index('Date')

confirmed_df_t_sl_ts = confirmed_df_t_sl_ts.drop(['Value'], axis=1)

#confirmed_df_t_sl_ts = confirmed_df_t_sl_ts.cumsum()

plt.plot(confirmed_df_t_sl_ts)

plt.show()
Original = confirmed_df_t_sl_ts

rolmean = confirmed_df_t_sl_ts.rolling(50).mean()

rolstd = confirmed_df_t_sl_ts.rolling(14).std()

rolmean7 = confirmed_df_t_sl_ts.rolling(20).mean()

plt.figure(figsize=(20,10))

plt.plot(confirmed_df_t_sl_ts, color='blue',label='Original')

plt.plot(rolmean, color='red', label='Rolling Mean')

plt.plot(rolmean7, color='yellow', label='Rolling Mean')

plt.legend(loc='best')

plt.title('Rolling Mean & Standard Deviation')



plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(Original)



trend = decomposition.trend

seasonal = decomposition.seasonal

residual = decomposition.resid



plt.subplot(411)

plt.plot(Original, label="Original")

plt.legend(loc='best')

plt.subplot(412)

plt.plot(trend, label="Trend")

plt.legend(loc='best')

plt.subplot(413)

plt.plot(seasonal, label="Seasonal")

plt.legend(loc='best')

plt.subplot(414)

plt.plot(residual, label="Residual")

plt.legend(loc='best')
from fbprophet import Prophet

cap = 300#china

#cap = 4.518*(10**9)

floor = 0
df = residual.reset_index()

df.columns = ['ds', 'y']

# df['cap'] = cap

# df['floor'] = floor

m_residual = Prophet()

m_residual.fit(df)

future_residual = m_residual.make_future_dataframe(periods=60)

# future_residual['cap'] = cap

# future_residual['floor'] = floor

future_residual = m_residual.predict(future_residual)

forcasted_residual = future_residual[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forcasted_residual.ds = pd.to_datetime(forcasted_residual.ds)

forcasted_residual = forcasted_residual.set_index('ds')

plt.figure(figsize=(50,10))

plt.plot(forcasted_residual.index, forcasted_residual.yhat)
df = trend.reset_index()

df.columns = ['ds', 'y']

df['cap'] = cap

df['floor'] = floor

m_trend = Prophet(growth='logistic')

m_trend.fit(df)

future_trend = m_trend.make_future_dataframe(periods=60)

future_trend['cap'] = cap

future_trend['floor'] = floor

future_trend = m_trend.predict(future_trend)

forcasted_trend = future_trend[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forcasted_trend.ds = pd.to_datetime(forcasted_trend.ds)

forcasted_trend = forcasted_trend.set_index('ds')

plt.figure(figsize=(50,10))

plt.plot(forcasted_trend.index, forcasted_trend.yhat)
df = seasonal.reset_index()

df.columns = ['ds', 'y']

# df['cap'] = cap

# df['floor'] = floor

m_seasonal = Prophet()

m_seasonal.fit(df)

future_seasonal = m_seasonal.make_future_dataframe(periods=60)

# future_seasonal['cap'] = cap

# future_seasonal['floor'] = floor

future_seasonal = m_seasonal.predict(future_seasonal)

forcasted_seasonal = future_seasonal[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

forcasted_seasonal.ds = pd.to_datetime(forcasted_seasonal.ds)

forcasted_seasonal = forcasted_seasonal.set_index('ds')

plt.figure(figsize=(50,10))

plt.plot(forcasted_seasonal.index, forcasted_seasonal.yhat)
x1 = forcasted_residual.merge(forcasted_trend, on='ds')

x2 = x1.merge(forcasted_seasonal, on='ds')

forcasted_original = pd.DataFrame()

forcasted_original["predicted_cases"] = x2[['yhat_x','yhat_y','yhat']].sum(axis=1)

forcasted_original["min"] = x2[['yhat_lower_x','yhat_lower_y','yhat_lower']].sum(axis=1)

forcasted_original["max"] = x2[['yhat_upper_x','yhat_upper_y','yhat_upper']].sum(axis=1)

forcasted_original.tail()
import datetime

today = datetime.date.today()

today = today.strftime("%Y-%m-%d")

forcasted_original.loc[forcasted_original.index==today]