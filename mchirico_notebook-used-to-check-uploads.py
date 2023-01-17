import pandas as pd

import numpy as np

import datetime





import warnings

warnings.filterwarnings("ignore")

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", color_codes=True)





dateparse = lambda x: datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S')



# Read data 

d=pd.read_csv("../input/911.csv",

    header=0,names=['lat', 'lng','desc','zip','title','timeStamp','twp','addr','e'],

    dtype={'lat':str,'lng':str,'desc':str,'zip':str,

                  'title':str,'timeStamp':str,'twp':str,'addr':str,'e':int}, 

     parse_dates=['timeStamp'],date_parser=dateparse)





# Set index

d.index = pd.DatetimeIndex(d.timeStamp)

d=d[(d.timeStamp >= "2016-01-01 00:00:00")]
d['timeStamp'].max(),d['timeStamp'].min()
d['day']=d['timeStamp'].dt.day

d['month']=d['timeStamp'].dt.month

d['date']=d['timeStamp'].dt.date

d['hr']=d['timeStamp'].dt.hour
min(d['date'].value_counts())
t=d[(d.timeStamp >= "2017-03-08 16:00:00") & (d.timeStamp <= "2017-03-08 23:00:00")]

t['hr'].value_counts()

t=d[(d.timeStamp >= "2017-03-01 16:00:00") & (d.timeStamp <= "2017-03-08 23:00:00")]

t=t[(t['hr']==21)]



t.groupby(['hr','date']).size()
d['lat'].max(),d['lat'].min(),d['lng'].max(),d['lng'].min()
d[(d['title']=='EMS: ACTIVE SHOOTER')]
#Fire: TRAIN CRASH 

d[(d['title']=='Fire: TRAIN CRASH')]
# EMS: PLANE CRASH    

d[(d['title']=='EMS: PLANE CRASH')]