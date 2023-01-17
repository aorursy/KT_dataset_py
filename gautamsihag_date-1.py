import numpy as np
import pandas as pd 
from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))
import datetime

train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=1000)
train1 = pd.read_csv("../input/train.csv", parse_dates=['date_time'], nrows=10)
train1['srch_ci']
#df['year'] = df['ArrivalDate'].dt.year
#df['year'] = pd.DatetimeIndex(df['ArrivalDate']).year
train1['month'] = pd.DatetimeIndex(train1['date_time']).month
train1['year'] = train1['date_time'].dt.year
train1['hour'] = train1['date_time'].dt.hour
train1[['year','hour','month']]
train1.ix[(train1['hour'] >= 5) & (train1['hour'] <= 10), 'hour'] = 2
train1['hour']
train1.ix[train1.hour >= 5,'year'] = 1
train1['year']
t = pd.tslib.Timestamp.now()
t
t.to_datetime()
t.hour
t.date()
t.dayofweek
t.day
t.year
t.time()
t.month
a=[]
a.append(int(t.month))
a.append(int(t.year))
a
len(a)