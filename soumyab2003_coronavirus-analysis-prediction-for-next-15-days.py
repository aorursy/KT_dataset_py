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
import folium
data=pd.read_csv("../input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")
data.head()
data['Province/State'].isnull().sum()
data.Date.unique()
data=data.replace(to_replace ="2020-01-02 23:00:00", 

                 value ="2020-02-01 23:00:00")
data.Date.unique()
data.dtypes
data['Last Update']= pd.to_datetime(data['Last Update']) 
#19-nCoV, 31 January 2020**")

from datetime import date

data_final=data[(data['Last Update'] >= '2020-2-2 00:00:00') & (data['Last Update'] < '2020-3-2 00:00:00')]
data_final=data_final.drop(["Sno"],axis=1)
data_final=data_final.reset_index(drop=True)
data_final.head()
data.Country.unique()
data=data.replace(to_replace ="China", 

                 value ="Mainland China")
data.Country.unique()
print('Total Confirmed Cases:',data_final['Confirmed'].sum())

print('Total Deaths: ',data_final['Deaths'].sum())

print('Total Recovered Cases: ',data_final['Recovered'].sum())
data.dtypes
from datetime import date
data_final.shape
#data_final=data_final.drop(["Date"],axis=1)
countries=data_final.groupby(["Country"]).sum().reset_index()
countries
countries.head()
top10_countries=countries.nlargest(10,['Confirmed']).reset_index(drop=True)
top10_countries.head()
top10_countries.plot('Country',['Confirmed'],kind = 'bar',figsize=(20,30))
data_china=data_final[data_final['Country']=="Mainland China"]
data_china.head()
data_china['Province/State'].isnull().sum()
data_china=data_china.groupby(["Province/State"]).sum()
data_china.reset_index()
data_china10=data_china.nlargest(10,['Confirmed']).reset_index()
data_china10.head()
from matplotlib import pyplot as plt
#fig = plt.figure(figsize=(10,))

data_china10.plot('Province/State',['Confirmed'],kind = 'bar',figsize=(20,10))



data_china10.plot('Province/State',['Deaths'],kind = 'bar',figsize=(10,10))
data.head()
data_trend=data[["Date","Confirmed","Deaths","Recovered"]]
data_trend.head()
data_trend=data_trend.groupby(["Date"]).sum().reset_index()
data_trend
trend=data[["Date","Confirmed","Deaths","Recovered"]]
trend.head()
trend=trend.groupby(["Date"]).sum().reset_index()
trend.plot('Date',['Confirmed'],figsize=(10,10))
trend.plot('Date',['Deaths'],figsize=(10,10))
trend.plot('Date',['Recovered'],figsize=(10,10))
import pandas as pd

import datetime

import folium

from folium.map import *

from folium import plugins

from folium.plugins import MeasureControl

from folium.plugins import FloatImage
data.head()
Nonconfirmed_cases=data.loc[data['Confirmed']==0.0] 
len(Nonconfirmed_cases)
Nonconfirmed_cases
data.head()
data1=data[['Date','Country','Confirmed','Deaths','Recovered']]
data1.head()
data1.dtypes
data1['Date']= pd.to_datetime(data1['Date']) 
data1.head(10)
data1=data1.groupby(["Date"]).sum().reset_index()
data1
data1.plot('Date',['Confirmed'],figsize=(10,10))
data1=data1.drop(['Deaths','Recovered'],axis=1)
data1
data1.set_index('Date',inplace=True)

data1=data1.diff()
data1=data1.fillna(555)

data1.reset_index(inplace=True)
data1
data1.columns = ['ds', 'y']

data1
from fbprophet import Prophet

m = Prophet()

m.fit(data1)

future = m.make_future_dataframe(periods=15)

forecast = m.predict(future)

forecast
print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-data1['y'])**2)) )
future.tail()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
m.plot(forecast,

              uncertainty=True)
m.plot_components(forecast)
d1=forecast[['ds','yhat']]
d1.head()
d1['total predicted cases']=d1['yhat'].cumsum()
d1=d1.drop(['yhat'],axis=1)
d1.head()
d1.plot('ds',['total predicted cases'],figsize=(10,10))