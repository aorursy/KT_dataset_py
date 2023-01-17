# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install DataScienceHelper
!pip install --upgrade pip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import DataScienceHelper as dsh
import plotly.express as px
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

%matplotlib inline
import time
from datetime import datetime
import re
from math import *
data = pd.read_csv("/kaggle/input/electricity-consumption/train.csv")
data.head()
data.tail()
data.isnull().sum()
data.describe()
data.count()
data.info()
data.memory_usage()
data.windspeed.value_counts()
plott = data.windspeed
plt.plot(plott)
plt.xlabel("samples")
plt.ylabel("frequency of windspeed")
plt.title("windspeed")
plt.show()
plt.scatter(data.windspeed,data.electricity_consumption,c='green')
plt.xlabel("frequecy of windspeed")
plt.ylabel("electricty consumption")
plt.title("windspeed distribution")
plt.show()
average = round(data.windspeed.mean(),3)
max_windspeed = round(max(data.windspeed),3)
min_windspeed = round(min(data.windspeed),3) 
print(f'The average windspeed is : {average} ')
print(f'The maximum windspeed is : {max_windspeed}')
print(f'The minimum windspeed is : {min_windspeed}')
avg_pressure = round(data.pressure.mean(),3)
max_pressure = round(data.pressure.max(),3)
min_pressure = round(data.pressure.min(),3)
print(f'The average pressure is : {avg_pressure}')
print(f'The maximum pressure is : {max_pressure}')
print(f'The minimum pressure is : {min_pressure}')
plt.plot(data.pressure)
plt.xlabel("samples")
plt.ylabel("frequency of pressure")
plt.title("Pressure Distribution")
plt.show()
plt.scatter(data.pressure,data.electricity_consumption,c='red')
plt.xlabel("frequency of pressure")
plt.ylabel("electricity consumption")
plt.title("Pressure Distribution")
plt.show()
plt.scatter(data.pressure,data.windspeed,c='blue')
plt.xlabel("frequency of pressure")
plt.ylabel("frequency of windspeed")
plt.title("Pressure Distribution")
plt.show()
sns.countplot(x='var2',data = data)
fig,ax = plt.subplots(figsize = (15,10))
corr = data.corr()
sns.heatmap(corr,xticklabels = corr.columns,annot = True,yticklabels = corr.columns,linewidth =1.2)
corr[abs(corr['electricity_consumption']) > 0.1]['electricity_consumption']
data.var1.value_counts()
data.var2.value_counts()
data.describe().T
data.head().T
data1 = (data
         .set_index(pd.to_datetime(data.datetime))
         .sort_index()
         .rename(columns={'var1':'dewpnt','var2':'peak_deamand'})
         .drop(columns = ['ID','datetime'])
        )
data1.head().T
fig = plt.figure(figsize=(18,10))
axis = fig.gca()
axis.plot(data1.electricity_consumption)
axis.set_title('Total City Electricity Consumption ')
axis.set_ylabel('Power (kWatt)')
axis.set_xlabel('Datetime')
plt.show()
fig = plt.figure(figsize=(18,10))
plt.title("Total city electrity consumption")
plt.xlabel("Datetime")
plt.ylabel("Power (Kwatt/hr)")
plt.plot(data1.electricity_consumption)
plt.show()
fig = data1['electricity_consumption'].plot.hist(figsize=(13,7),title="electricity consumption freq",bins= 100)
plt.xlabel("Power (Kwatt/hrs)")
plt.show()
data1 = (data1.assign(dayofweek = data1.index.week ,
                      year = data1.index.year,
                      month = data1.index.month,
                      day = data1.index.day,
                      week = data1.index.week,
                      week_day = data1.index.weekday,
                      quarter = data1.index.quarter,
                      hours = data1.index.hour                      
))
data1.head()
data1.tail()
fig = (data1['electricity_consumption'].loc[(data1['electricity_consumption'].index >= '2013-11-01') & 
                                            (data1['electricity_consumption'].index < "2014-03-01") ] .plot(figsize = (15,10),title='Energy Consumption in Winter of 2013-14'))
plt.show()

fig = (data1['electricity_consumption'].loc[(data1['electricity_consumption'].index >= '2013-03-01') & 
                                            (data1['electricity_consumption'].index < "2014-07-01") ] .plot(figsize = (15,10),title='Energy Consumption in Summer of 2014'))
plt.show()

fig = (data1['electricity_consumption'].loc[(data1['electricity_consumption'].index >= '2014-11-01') & 
                                            (data1['electricity_consumption'].index < "2015-03-01") ] .plot(figsize = (20,10),title='Energy Consumption in Winter of 2013-14'))
plt.show()

fig = (data1['electricity_consumption'].loc[(data1['electricity_consumption'].index >= '2013-11-01') & 
                                            (data1['electricity_consumption'].index < "2013-12-01") ] .plot(figsize = (15,10),title='Energy Consumption in November 2013'))
plt.show()

fig = (data1['electricity_consumption'].loc[(data1['electricity_consumption'].index >= '2016-11-01') & 
                                            (data1['electricity_consumption'].index < "2016-12-01") ] .plot(figsize = (15,10),title='Energy Consumption in November 2016'))
plt.show()
data1.head()
fig =data1.pivot_table(index = data1['hours'],
                       columns = 'week_day',
                       values = 'windspeed',
                       aggfunc = 'mean').plot(figsize = (15,10),title = 'windspeed-dailytrend')
plt.ylabel("windspeed")
plt.show()
data1 = data1.replace({"week_day" :{0 :"Monday",1 :"Tuesday",2 : "Wednesday",3 : "Thursday",4 :"Friday",5 :"Saturday",6 :"Sunday" }})
data1.head()
columns = list(data1._get_numeric_data().keys())
dsh.show_kdeplot(data1,columns)
columns = list(data1._get_numeric_data().keys())
dsh.show_boxplot(data1,columns)