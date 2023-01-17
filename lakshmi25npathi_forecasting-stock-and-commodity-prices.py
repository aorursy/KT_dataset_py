# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Time series analysis

import datetime

from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')

wt_df=pd.read_csv('../input/websitetraffic/website-traffic.csv')

wt_df.head()



wt_df['date_of_visit']=pd.to_datetime(wt_df.MonthDay.str.cat(wt_df.Year.astype(str),sep=' '))

wt_df.head(10)
#Time series plot

wt_df.plot(x='date_of_visit',y='Visits',title='Website visitors per day')

#Time series components(seasonality(These are the periodic fluctuations in the observed data),trend(This is the increasing or decreasing behavior of the series with time),

#residual(This is the remaining signal after removing the seasonality and trend

#signals))

from statsmodels.tsa.seasonal import seasonal_decompose

#Extract visits as a series from dataframe

ts_visits=pd.Series(wt_df.Visits.values,index=pd.date_range(wt_df.date_of_visit.min(),wt_df.date_of_visit.max(),freq='D'))

decompose=seasonal_decompose(ts_visits.interpolate(),freq=24)

decompose.plot()
#Smoothing techniques(it helps reduce the effect of random variation and

#clearly reveal the seasonality, trend, and residual components of the series)

#Moving average

wt_df['moving_average']=wt_df['Visits'].rolling(window=3,center=False).mean()

wt_df.fillna(0).head(10)
plt.plot(wt_df.Visits,'-',color='black',alpha=0.3)

plt.plot(wt_df.moving_average,color='b')

plt.title('Website Visiting and Moving Average Smoothing')

plt.legend()

plt.show()
#Exponetial smoothing(exponentially weighted moving average)(EWMA)

wt_df['ewma']=wt_df['Visits'].ewm(halflife=3,ignore_na=False,min_periods=0,adjust=True).mean()

wt_df.fillna(0).head(10)

plt.plot(wt_df.Visits,'-',color='b',alpha=0.3)

plt.plot(wt_df.ewma,color='g')

plt.title('Website visit and Exponential Smoothing')

plt.legend()

plt.show()
#Forecasting Gold price

gold_df=pd.read_csv('../input/gold-price/BSE-BOM590111.csv')



gold_df=gold_df.rename(columns={'Total Turnover':'Total_Turnover'})

gold_df.head()
#ARIMA(Auto Regressive Integrated Moving Average) model



gold_df.plot(x='Date',y='Total_Turnover',figsize=(15,6))

plt.show()
new_df=gold_df[['Date','Total_Turnover']]

new_df.head()

print(new_df.shape)

new_df.head().fillna(0)
#Dicky Fuller test



adfuller(new_df.Total_Turnover.values)

#Rolling statistics

new_df['Rolling_Mean']=new_df['Total_Turnover'].rolling(window=5,center=False).mean()

new_df['Rolling_Std']=new_df['Total_Turnover'].rolling(window=5,center=False).std()

Rolling_Mean=np.log(new_df.Rolling_Mean)

Rolling_Std=np.log(new_df.Rolling_Std)



plt.plot(Rolling_Mean,'-',color='b',alpha=0.5)

plt.plot(Rolling_Std,color='g')

plt.title('Rolling Statistics')

plt.legend()

plt.show()