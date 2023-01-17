%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np

def dateTimeMaker(times):
    return datetime.datetime(year=2000, month=1, day=1) + datetime.timedelta(minutes=int(times))
# 01-01-2000 is an arbitrary date
%%time
df= pd.read_csv('../input/train.csv')
df['datetime']=df['time'].apply(dateTimeMaker)
df['date']=df['datetime'].dt.date
df['minutes']=df['datetime'].dt.minute
df['hours']=df['datetime'].dt.hour
df['weekday']=df['datetime'].dt.weekday
f, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=False)
for i, param in enumerate('date,minutes,hours,weekday'.split(',')):
    row,col=divmod(i,2)[0],divmod(i,2)[1]
    grouped = pd.groupby(df[['accuracy',param]],by=param)[['accuracy']].mean()
    axes[row,col].plot(grouped.index, grouped['accuracy'])
    axes[row,col].set_ylim([40, 100]) 
    axes[row,col].set_xlabel(param,size=15)
    axes[row,0].set_ylabel("accuracy",size=15)