import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import os
print(os.listdir("../input/"))
#The dataset contains details of the number of visitors to Australia and Japan quarterly
data=pd.read_csv('../input/visitors.csv')
data.head()
data.dtypes
#Set date column as an Index and Convert to Datetime
data=pd.read_csv('../input/visitors.csv',parse_dates=['Date'],index_col='Date')
data.head()
#pd.to_datetime(data['Date']).head()
#data.set_index('Date')
data.index
# Data of specific year
data['1999']
#Data of specific month of the year
data['1999-10']
#Data from one date to another
data['2001-01-01':'2005-01-01']
data['Australia'].resample('Y').mean()
data['Australia'].plot()
# Let's delete the datetime information
data.reset_index(inplace=True)
data=data.drop('Date',axis=1)
data.head()
#Freq='B' B-'business day frequency' (weekend excluded)
dt_range=pd.date_range(start="3/1/2017",end="5/15/2017",freq='B')
dt_range
dt_range.shape
#set new datetime as an Index
data.set_index(dt_range,inplace=True)
data.head()
#D-calendar day frequency All days included
newdt_range=pd.date_range(start="3/1/2017",end="5/15/2017",freq='D')
newdt_range
##Weekend dates or dates which are not included (missing)
newdt_range.difference(data.index)
print(data.head())
data.index.dtype
data.index.freq
#http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
#Data will be forwared to next day same as previous day as we have specified freq-'D' and method forward fill
data.asfreq('D',method='ffill').head(5)
#H-hourly frequency and method-forward fill
data.asfreq('H',method='ffill').head()
data={
    'year':[2018,2019],
    'month':[3,9],
    'day':[1,11]   
}
data
pd.to_datetime(data)
#Custom date time format
pd.to_datetime('2017^01^01',format='%Y^%m^%d')
pd.to_datetime(['2001*05*21','2001*05*3'],format='%Y*%m*%d')
pd.to_datetime(['jan 1,2001','1/1/2010','hello'],errors='raise')
pd.to_datetime(['jan 1,2001','1/1/2010','hello'],errors='ignore')
pd.to_datetime(['jan 1,2001','1/1/2010','hello'],errors='coerce')
#NaT :-invalid parsing will be set as NaT
year=pd.Period('2017')
year
#Frequency is 'A' for end of year frequency
year.is_leap_year
year.start_time
year.end_time
#3 years later
year+3
month=pd.Period('2001-9')
month
month.start_time
month.end_time
# 3 months back
month-3
#Daily Period
day=pd.Period('2018-03-09')
day
day.start_time
day.end_time
#9 days later
day+9
hour=pd.Period('2018-03-09 12')
hour
#5 hours later
hour+5
quarter=pd.Period('2009Q1')
quarter
quarter.start_time
quarter.end_time
#change the quarter start
quarter=pd.Period('2009Q1',freq='Q-FEB')
quarter
quarter.start_time
quarter.end_time
week=pd.Period('2018-09-09',freq='W')
week
week=pd.Period('2018-09-11',freq='W')
week
#D:calendar day frequency
quarter=pd.Period('2009Q1',freq='Q-FEB')
quarter.asfreq('D',how='start')
quarter.asfreq('D',how='end')
range_m=pd.period_range('2009','2011',freq='M')
range_m
range_q=pd.period_range('2009','2011',freq='q')
range_q
#to_timestamp
range_q_ts=range_q.to_timestamp()
range_q_ts
#to_period
range_q_period=range_q_ts.to_period()
range_q_period
#pytz brings the Olson tz database into Python. 
#This library allows accurate and cross platform timezone calculations using Python 2.4 or higher
import pytz
pytz.all_timezones[:5]
dt=pd.DatetimeIndex(start='2010-09-09 03:00',freq='H',periods=9,tz='Australia/Sydney')
dt
#Convert to other time zone - tz_convert
dt.tz_convert('Asia/Calcutta')
dt=pd.date_range(start='2017-01-01 09:00',periods=9,freq='15min')
print(dt.tz)
dt
# DatetimeIndex in Asia/Calcutta time zone:
dt=dt.tz_localize(tz='Asia/Calcutta')
dt
data=pd.read_csv('../input/visitors.csv',parse_dates=['Date'],index_col='Date',nrows=5)
data
# shift 1 place forward (positive)
data.shift(periods=1)
#shift 3 places backward (negative)
data.shift(periods=-3)
#Find the Differnce of visitors - previous day and today
#data['Aus-visitors-today']=data['Australia']
#data['Aus-visitors-previous day']=data['Australia'].shift(periods=1)
#data['Differnce of visitors']=data['Aus-visitors-today']-data['Aus-visitors-previous day']
#data['Differnce of visitors']
data
data['Australia']
 #shift the date (next quarter date)
data['Australia'].tshift(1)
data['Australia'].tshift(-1)
#refer :- https://pandas.pydata.org/pandas-docs,codebasics to learn more