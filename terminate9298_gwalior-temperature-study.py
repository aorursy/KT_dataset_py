# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime  import datetime

import time

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
temp = pd.read_csv('../input/VIGR_updated.txt',header=0,low_memory = False)
temp['date_time'] = pd.to_datetime(temp['valid'])

# temp['temp_year'],temp['temp_month'],temp['temp_month_day'],temp['temp_hour'] ,temp['temp_minute'],_,_,temp['year_day'],_  = temp['valid'].apply(lambda x: time.strptime(x, '%Y-%m-%d %H:%M')) 

temp['temp_year']=temp['date_time'].dt.year

temp['temp_month']=temp['date_time'].dt.month

temp['temp_day']=temp['date_time'].dt.day

temp['temp_hour']=temp['date_time'].dt.hour

temp['temp_minute']=temp['date_time'].dt.minute

temp['temp_date']=temp['date_time'].dt.date

temp['temp_time']=temp['date_time'].dt.time





temp.columns
display((temp.isnull().sum()/temp.shape[0])*100)

temp_updated = temp.drop(columns=['station' , 'wxcodes' , 'peak_wind_gust_mph',

       'peak_wind_drct', 'peak_wind_time','p01i' , 'metar','gust_mph'] , axis=1)
temp_updated.columns
temp_update_new = temp_updated.loc[:65460,]

display(temp_update_new.tail(10).T)

display(temp_updated.info())
# Data is too much Varable

display(temp['temp_date'].value_counts().value_counts())

display(temp['temp_time'].value_counts())
temp_new = temp[temp['temp_year'] <= 2016]

temp_test = temp[temp['temp_year'] > 2016]
temp_new.tail(5)
plt.figure(figsize=(18,6))

temp_new['tmpc'].plot()

plt.show()
temp_temp = temp_new.sort_values(by='tmpc',ascending=1)
temp_temp.head()