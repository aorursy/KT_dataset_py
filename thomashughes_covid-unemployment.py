#Author: Thomas Hughes

#Date: June 9 2020

#Description: An analysis of the unemployment rate during the covid-19 crisis.

#Data Sources:

# - Bureau of Labor Statistics (BLS)
!pip install pmdarima

import numpy as np #General data analysis

import pandas as pd #General data analysis

from statsmodels.tsa.seasonal import seasonal_decompose #time series and forecasting

from statsmodels.tsa.statespace.sarimax import SARIMAX #time series and forecasting

from pmdarima import auto_arima #time series and forecasting

from statsmodels.tools.eval_measures import rmse

import matplotlib.pyplot as plt

import datetime as dt

%matplotlib inline

import json

import requests

plt.rc('figure',figsize=(12,5))

plt.rc('font',size=13)
def create_Date(y,m):

    '''

    Transforms date information in unemp_df to datetime

    object.

    '''

    return dt.datetime.strptime(str(y)+m[1:3]+'01','%Y%m%d').date()
#We start by using the BLS API to get the

#US unemployment rate not seasonally adjusted.

url='https://api.bls.gov/publicAPI/v1/timeseries/data/'

headers={'Content-type':'application/json'}

data = json.dumps({'seriesid':['LNU04000000'],'startyear':'2011','endyear':'2021'})

raw = requests.request(method='POST',url=url,headers=headers,data=data)
#We will save the results to a json file so that we can read into a pandas data frame.

json.dump(raw.json()['Results']['series'][0]['data'],open('unemp_data.json','w'))

unemp_df = pd.read_json('unemp_data.json')

#Use function to create date column in unemp_df

unemp_df['date']=np.vectorize(create_Date)(unemp_df['year'],unemp_df['period'])

#Set the date column to be the index.

unemp_df.set_index(['date'],inplace=True)

#Drop irrelevant columns

unemp_df.drop(['year','period','periodName','latest','footnotes'],axis=1,inplace=True)

#unemp_df is ordered newest to oldest, so reorder unemp_df oldest to newest

unemp_df = unemp_df.iloc[::-1]

#add a frequency to the timestamps

unemp_df.index.freq = "MS"
unemp_df.describe()
unemp_df['value'].plot()
unemp_df.tail()
unemp_decompose = seasonal_decompose(unemp_df['value'][unemp_df.index<dt.datetime.strptime('2020-03-01','%Y-%m-%d').date()],period=3,model='additive');

unemp_decompose.plot();
unemp_decompose.resid.hist(bins=10)
auto_arima(unemp_df['value'][:len(unemp_df)-3],seasonal=True,m=3).summary()
train = unemp_df.iloc[:len(unemp_df)-14]

train
test = unemp_df.iloc[len(unemp_df)-14:-3]

test
# create the model

model = SARIMAX(train,order=(2,1,1),seasonal_order=(1, 0, 1, 3))

#fit the model

results = model.fit()

#display a summary of model statistics

results.summary()
#Now use the model to predict into the test data

start = len(train)

end = len(train) + len(test) - 1

predictions = results.predict(start,end,typ='levels').rename('SARIMA predictions') #typ='levels' to avoid problems in differencing
#plot the predictions against the test data to assess performance

test.plot(legend=True,figsize=(16,12))

predictions.plot(legend=True)
#Check to see the root of the mean squared error.

error = rmse(test['value'],predictions)

error
#Create the forecast model

fmodel = SARIMAX(unemp_df[:-3],order=(2,1,1),seasonal_order=(1, 0, 1, 3))
#Fit the model and print out model statistics

fresults = fmodel.fit()

fresults.summary()
#Use the model to create the forecast

fcast = fresults.predict(len(unemp_df[:-3]),len(unemp_df[:-3])+12,typ='levels').rename('SARIMA forecast')

#Plot the forecast against the actual values

unemp_df['value'].plot(legend=True,figsize=(12,8))

fcast.plot(legend=True)
fcast
unemp_increase = pd.DataFrame({'Actual And Fed':[4.5,14.4,13.0,9.3],'SARIMA Forecast':[3.493709,3.111710,3.168717,3.168041]},index=['2020-03-01','2020-04-01','2020-05-01','2020-12-01'])

unemp_increase['Percent Increase'] = ((unemp_increase['Actual And Fed'] - unemp_increase['SARIMA Forecast'])/unemp_increase['SARIMA Forecast'])*100

unemp_increase
unemp_increase['Percent Increase'].plot(legend=True)