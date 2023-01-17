#Loading packages and Data



from IPython.display import Image

import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

from datetime import datetime

from random import random

from math import sqrt

from numpy import concatenate

from numpy import array

import matplotlib.pyplot as plt



from statsmodels.tsa.stattools import grangercausalitytests

from statsmodels.tsa.vector_ar.var_model import VAR

from statsmodels.tsa.stattools import adfuller

from statsmodels.tsa.vector_ar.vecm import coint_johansen

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from statsmodels.tsa.seasonal  import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



!pip install pyramid-arima

from pyramid.arima import auto_arima





#!pip install plotly==3.10.0



from fbprophet import Prophet

#from plotly.plotly import plot_mpl





from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor







from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pd.plotting.register_matplotlib_converters()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#Loading csv

data=pd.read_csv('../input/smart-home-dataset-with-weather-information/HomeC.csv')
data.head()
data.info()
# removing the truncated record

data=data[:-1]

data.shape
#new daterange in increments of minutes

time_index = pd.date_range('2016-01-01 05:00', periods=len(data),  freq='min')  

time_index = pd.DatetimeIndex(time_index)

data['time']=time_index
#changing column names before doing some calculation as they look weird with "[kw]"

data.columns=['time', 'use', 'gen', 'House overall', 'Dishwasher',

       'Furnace 1', 'Furnace 2', 'Home office', 'Fridge',

       'Wine cellar', 'Garage door', 'Kitchen 12',

       'Kitchen 14', 'Kitchen 38', 'Barn', 'Well',

       'Microwave', 'Living room', 'Solar', 'temperature',

       'icon', 'humidity', 'visibility', 'summary', 'apparentTemperature',

       'pressure', 'windSpeed', 'cloudCover', 'windBearing', 'precipIntensity',

       'dewPoint', 'precipProbability']
data['gen'].head()
data['Solar'].head()
(data['gen']-data['Solar']).value_counts()
data=data.drop('gen',axis=1)
(data['House overall']-data['use']).value_counts()
data=data.drop('House overall',axis=1)
#getting  hour, day,week, month from the date column

data['day']= data['time'].dt.day

data['month']= data['time'].dt.month

data['week']= data['time'].dt.week

data['hour']= data['time'].dt.hour
import seaborn as sns

def visualize(label, cols):

    fig,ax=plt.subplots(figsize=(14,8))

    colour= ['red','green','blue','yellow']

    for colour,col in zip(colour,cols):

            data.groupby(label)[col].mean().plot(ax=ax,label=col,color=colour)

    plt.legend()



visualize('hour',['Furnace 1','Furnace 2'])
visualize('day',['Furnace 1','Furnace 2'])
visualize('month',['Furnace 1','Furnace 2'])
data['Furnace']= data['Furnace 1']+data['Furnace 2']

data=data.drop(['Furnace 1','Furnace 2'], axis =1)
visualize('month',['Kitchen 12','Kitchen 14','Kitchen 38'])
visualize('week',['Kitchen 12','Kitchen 14','Kitchen 38'])
visualize('day',['Kitchen 12','Kitchen 14','Kitchen 38'])
visualize('hour',['Kitchen 12','Kitchen 14','Kitchen 38'])
data['Kitchen 38'].describe()
fig,ax=plt.subplots(2,2,figsize=(15,10))



data.groupby('hour')['Kitchen 38'].mean().plot(ax=ax[0,0],color='green',label= 'kitchen 38')

data.groupby('day')['Kitchen 38'].mean().plot(ax=ax[0,1],color='green',label= 'kitchen 38')

data.groupby('week')['Kitchen 38'].mean().plot(ax=ax[1,0],color='green',label= 'kitchen 38')

data.groupby('month')['Kitchen 38'].mean().plot(ax=ax[1,1],color='green',label= 'kitchen 38')



                                                     



plt.legend()
data['icon'].value_counts()
data['summary'].value_counts()
data.groupby('summary')['Solar'].sum()
data=data.drop(['icon','summary'], axis =1)
data['cloudCover'].dtypes
data['cloudCover'].head()
data['cloudCover'].value_counts()
data['cloudCover'].unique()
data['cloudCover'].replace(['cloudCover'], method='bfill', inplace=True)

data['cloudCover'].unique()
data['cloudCover']=data['cloudCover'].astype('float')
data.info()
data.index= data['time']

#daily resampling

dataD=data.resample('D').mean()
dataD.info()
#hourly resampling

dataH=data.resample('H').mean()
weathercols= ['temperature', 'humidity','visibility', 'apparentTemperature', 'pressure', 'windSpeed',

       'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint','precipProbability']

Housecols = ['Dishwasher','Furnace', 'Home office', 'Fridge','Wine cellar', 'Garage door', 'Kitchen 12','Kitchen 14', 

             'Kitchen 38', 'Barn', 'Well','Microwave', 'Living room']

useweather=['use','temperature', 'humidity','visibility', 'apparentTemperature', 'pressure', 'windSpeed',

       'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint','precipProbability']

solarweather=['Solar','temperature', 'humidity','visibility', 'apparentTemperature', 'pressure', 'windSpeed',

       'cloudCover', 'windBearing', 'precipIntensity', 'dewPoint','precipProbability']

usesolar=['use','Solar']
# load dataset

def series_visualize(data, cols):

    dataset = data[cols]

    values = dataset.values

    # specify columns to plot    

    groups = [i for i in range(len(cols))]

    j = 1

# plot each column

    plt.figure(figsize=(18,13))

    for group in groups:

        plt.subplot(len(groups), 1, j)

        plt.plot(values[:, group])

        plt.title(dataset.columns[group], y=0.5, loc='right')

        j += 1

    plt.show()
#series_visualize(dataH,Housecols)

series_visualize(dataD,Housecols)
series_visualize(dataH,usesolar)
series_visualize(dataD,usesolar)
datause=dataD.iloc[:,0].values

#fig,ax=plt.subplots(figsize=(15,10))

plt.rcParams['figure.figsize'] = (14, 9)

seasonal_decompose(dataD[['use']]).plot()

result = adfuller(datause)

plt.show()
X= dataD.iloc[:,0].values

result = adfuller(X)

print('ADF Statistic: %f' % result[0])

print('p-value: %f' % result[1])

print('Critical Values:')

for key, value in result[4].items():

    print('\t%s: %.3f' % (key, value))
# split data into train and tests

train=dataD[dataD['month']<12].iloc[:,0]

test=dataD[dataD['month']>=12].iloc[:,0]

print("train has {} records, test has {} records".format(len(train),len(test)))
fig,ax=plt.subplots(figsize=(18,6))

train.plot(ax=ax)

test.plot(ax=ax)

plt.show()


# fit model withweekly seasonality 

model = ExponentialSmoothing(train.values,seasonal='add',seasonal_periods=7)

model_fit = model.fit()



# make prediction



y = model_fit.forecast(len(test))

y_predicted=pd.DataFrame(y,index=test.index,columns=['Holtwinter'])



plt.figure(figsize=(16,8))

plt.plot(test, label='Test')

plt.plot(y_predicted, label='Holtwinter')

plt.legend(loc='best')

plt.show()
rms = sqrt(mean_squared_error(test,y_predicted))

print(rms)
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf



# Draw Plot

fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)

x=plot_acf(train.tolist(), lags=50,ax=axes[0])

y=plot_pacf(train.tolist(), lags=50, ax=axes[1])

plt.show()
# first differencing

fig, axes = plt.subplots(1,2,figsize=(16,3), dpi= 100)

x=plot_acf(train.diff().dropna(), lags=50,ax=axes[0])

y=plot_pacf(train.diff().dropna(), lags=50, ax=axes[1])

plt.show()
def split_sequence(sequence, n_steps_in, n_steps_out):

    X, y = list(), list()

    for i in range(len(sequence)):

        # find the end of this pattern

        end_ix = i + n_steps_in

        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the sequence

        if out_end_ix > len(sequence):

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]

        X.append(seq_x)

        y.append(seq_y)

    return array(X), array(y)
# define input sequence

raw_seq = train[:307].values.tolist()

# choose a number of time steps

n_steps_in, n_steps_out = 28, 16

# split into samples

X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)

# reshape from [samples, timesteps] into [samples, timesteps, features]

n_features = 1

X = X.reshape((X.shape[0], X.shape[1], n_features))

# define model
#LSTM model

model = Sequential()

model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps_in, n_features)))

model.add(LSTM(100, activation='relu'))

model.add(Dense(n_steps_out))

model.compile(optimizer='adam', loss='mse')

# fit model

model.fit(X, y, epochs=50, verbose=0)
# demonstrate prediction

x_input = train[307:].values

x_input = x_input.reshape((1, n_steps_in, n_features))

yhat = model.predict(x_input, verbose=0)

print(yhat)
yhat=yhat.reshape(16,1)


forecast = pd.DataFrame(yhat,index = test.index,columns=['Prediction'])



#plot the predictions for validation set

plt.figure(figsize=(16,8))

plt.plot(test, label='Valid')

plt.plot(forecast, label='Prediction')

plt.show()
rms = sqrt(mean_squared_error(test,forecast))

print(rms)
f = m.plot_components(forecast)

predictions=pd.DataFrame(forecast[335:]['yhat'])

predictions.index=test.index



fig,ax=plt.subplots(figsize=(15,8))

test.plot(ax=ax)

predictions.plot(ax=ax)
rms = sqrt(mean_squared_error(test,forecast[['yhat']][335:]))

print(rms)