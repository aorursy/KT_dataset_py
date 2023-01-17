from __future__ import absolute_import, division, print_function, unicode_literals
# !pip install -q pmdarima
import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

# plt.style.use('fivethirtyeight')

import pathlib

import os

import seaborn as sns

import pandas as pd

from datetime import datetime



import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# import pmdarima as pm

from statsmodels.tsa.statespace.sarimax import SARIMAX

# import joblib

sns.set()
# from google.colab import drive

# drive.mount('/content/gdrive', force_remount=True)

# root_dir = "/content/gdrive/My Drive/"

# base_dir = root_dir + 'MachineLearning/DelhiTemperaturePrediction/'
# def displayDirContent(dir):

#   if pathlib.posixpath.exists(dir):

#     for name in list(pathlib.Path(dir).glob('*')):

#       print(name)

#   else:

#     print("Path does not exists")
# displayDirContent(base_dir)
print(os.listdir("../input/delhi-weather-data"))
data_dir = "../input/delhi-weather-data/"

# displayDirContent(data_dir)
data = pd.read_csv(data_dir + 'testset.csv')
data.head()
data.tail()
def overViewOfTheData(data,frows=5,lrows=5):

  print("Shape: ",data.shape,"\n\n")

  

  print("Columns: ",data.columns,"\n\n")



  print("Info : ")

  print(data.info())
overViewOfTheData(data)
plt.figure(figsize=(8,8))

sns.barplot(x=data.count()[:],y=data.count().index)

plt.xlabel('Non-Null Values Count')

plt.ylabel('Features')
data = data.drop([' _heatindexm',' _precipm',' _wgustm',' _windchillm'],axis=1)
# Date-Time column is not in the desired format. So, first we will convert it into the desired format (yyyy-mm-dd HH:MM)

# And the we will make that column the index of the data



data['datetime_utc'] = pd.to_datetime(data['datetime_utc'].apply(lambda x: datetime.strptime(x,"%Y%m%d-%H:%M").strftime("%Y-%m-%d %H:%M")))

data['datetime_utc'].head()
data = data.set_index('datetime_utc',drop=True)

data.index.name = 'datetime'
fig, ax = plt.subplots()

data[' _tempm'].plot(figsize=(15,12),ax=ax)

ax.set_xlabel('Date-Time')

ax.set_ylabel('Temperature in C')

ax.set_title('Temperature in Delhi')

plt.show()
# Dropping the data before 2001

data = data['2001':]
# We will remove the missing data and later we will interpolate the temperature for that missing data

print("Before : ", data.shape)

data.dropna(subset=[' _tempm'],inplace=True)

print("After :", data.shape)
data.index.minute.value_counts()
categoricalColumns = list(set(data.columns) - set(data._get_numeric_data().columns))

categoricalColumns
# We are resampling it by hours & filling the missing values using the interpolation method

# Notice here we will only get numeric columns so we will have to add the categorical columns additionaly

newdata = data.resample('H').mean().interpolate()

newdata.info()
# To resample the categorical data we will consider the firt observation and to fill the missing values we will use ffill method

newdata[list(categoricalColumns)] = data[categoricalColumns].resample('H').first().ffill().head()

newdata.head()
def plotAggregateValues(data,column=None):

  if column in data.columns:

    plt.figure(figsize = (18,25))

    

    ax1 = plt.subplot(4,2,1)

    newdata[column].groupby(newdata.index.year).mean().plot(ax=ax1,title='yearly mean values')

    ax1.set_xlabel('years')

    ax1.set_ylabel(column)

  

    ax2 = plt.subplot(4,2,2)

    newdata[column].groupby(newdata.index.month).mean().plot(ax=ax2,title='monthly mean values')

    ax2.set_xlabel('months')

    ax2.set_ylabel(column)



    # ax3 = plt.subplot(4,2,3)

    # newdata[column].groupby(newdata.index.weekday).mean().plot(ax=ax3,title='weekdays mean values')

    # ax3.set_xlabel('weekdays')

    # ax3.set_ylabel(column)



    ax4 = plt.subplot(4,2,4)

    newdata[column].groupby(newdata.index.hour).mean().plot(ax=ax4,title='hourly mean values')

    ax4.set_xlabel('hours')

    ax4.set_ylabel(column)



    # ax5 = plt.subplot(4,2,5)

    # newdata[column].groupby(newdata.index.minute).mean().plot(ax=ax5,title='minute wise mean values')

    # ax5.set_xlabel('minutes')

    # ax5.set_ylabel(column)



    # ax6 = plt.subplot(4,2,6)

    # newdata[column].groupby(newdata.index.second).mean().plot(ax=ax6,title='seconds wise mean values')

    # ax6.set_xlabel('seconds')

    # ax6.set_ylabel(column)



  else:

    print("Column name not specified or Column not in the data")
plotAggregateValues(newdata,' _tempm')
def plotBoxNdendity(data,col=None):

  if col in data.columns:    

    plt.figure(figsize=(18,8))



    ax1 = plt.subplot(121)

    data.boxplot(col,ax=ax1)

    ax1.set_ylabel('Boxplot temperature levels in Delhi', fontsize=10)



    ax2 = plt.subplot(122)

    data[col].plot(ax=ax2,legend=True,kind='density')

    ax2.set_ylabel('Temperature distribution in Delhi', fontsize=10)



  else:

    print("Column not in the data")
plotBoxNdendity(data,' _tempm')
train = newdata[:'2015']

test = newdata['2016':]
# Let's decompose the time series to visualize trend, season and noise seperately

def decomposeNplot(data):

  decomposition = sm.tsa.seasonal_decompose(data)



  plt.figure(figsize=(15,16))



  ax1 = plt.subplot(411)

  decomposition.observed.plot(ax=ax1)

  ax1.set_ylabel('Observed')



  ax2 = plt.subplot(412)

  decomposition.trend.plot(ax=ax2)

  ax2.set_ylabel('Trend')



  ax3 = plt.subplot(413)

  decomposition.seasonal.plot(ax=ax3)

  ax3.set_ylabel('Seasonal')



  ax4 = plt.subplot(414)

  decomposition.resid.plot(ax=ax4)

  ax4.set_ylabel('Residuals')



  return decomposition
# Resampling the data to mothly and averaging out the temperature & we will predict the monthly average temperature

ftraindata = train[' _tempm'].resample('M').mean()

ftestdata = test[' _tempm'].resample('M').mean()
# Taking the seasonal difference S=12 and decomposing the timeseries

decomposition = decomposeNplot(ftraindata.diff(12).dropna())
# Let's check for stationarity (Augmented Dickey Fuller test)

results = adfuller(ftraindata.diff(12).dropna())

results
# To get non-seasonal oreders of the SARIMAX Model we will first use ACF & PACF plots

plt.figure(figsize=(10,8))



ax1 = plt.subplot(211)

acf = plot_acf(ftraindata.diff(12).dropna(),lags=30,ax=ax1)



ax2 = plt.subplot(212)

pacf = plot_pacf(ftraindata.diff(12).dropna(),lags=30,ax=ax2)
# To get seasonal oreders of the SARIMAX Model we will first use ACF & PACF plots at seasonal lags 



lags = [12*i for i in range(1,4)]



plt.figure(figsize=(10,8))



ax1 = plt.subplot(211)

acf = plot_acf(ftraindata.diff(12).dropna(),lags=lags,ax=ax1)



ax2 = plt.subplot(212)

pacf = plot_pacf(ftraindata.diff(12).dropna(),lags=lags,ax=ax2)
model = SARIMAX(ftraindata,order=(0,0,1),seasonal_order=(0,1,1,12),trend='n')

results = model.fit()
# # Lets select the best model based on the aic & bic scores using auto_arima

# results = pm.auto_arima(ftraindata,

#                       seasonal=True, m=12,

#                       d=0,D=1,trace=True,

#                       error_action='ignore',

#                       suppress_warnings=True)
# Check the value of Prob(Q) if it is > 0.05 => The residuals are uncorrelated

# Similarly if Prob(JB) > 0.05 => The residuals are normally distributed

results.summary()
# Mean Absolute Error for training data

print(np.mean(np.abs(results.resid)))
diagnostics = results.plot_diagnostics(figsize=(10,10))
forecast = results.get_forecast(steps=len(ftestdata))
predictedmean = forecast.predicted_mean

bounds = forecast.conf_int()

lower_limit = bounds.iloc[:,0]

upper_limit = bounds.iloc[:,1]
plt.figure(figsize=(12,8))



plt.plot(ftraindata.index, ftraindata, label='train')

plt.plot(ftestdata.index,ftestdata,label='actual')



plt.plot(predictedmean.index, predictedmean, color='r', label='forecast')



plt.fill_between(lower_limit.index,lower_limit,upper_limit, color='pink')



plt.xlabel('Date')

plt.ylabel('Delhi Temperature')

plt.legend()

plt.show()
# displayDirContent(base_dir)
# filename = 'SARIMA_0_0_1_0_1_1_12.pkl'

# joblib.dump(results,filename = base_dir + 'Models/' + filename)