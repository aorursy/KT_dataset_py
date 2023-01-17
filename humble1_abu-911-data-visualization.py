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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns
sns.set()
data = pd.read_csv("../input/911.csv")
data.head()
data.shape
data.info()
column_names = list(data.columns)

column_names
data['timeStamp'] = pd.to_datetime(data['timeStamp'])

data['timeStamp'].head()
def type_reason(x):

    x = x.split(':')

    return x[0]
data["call_type"] = data["title"].apply(type_reason)

data.head()
def emergency_type_separator(x):

    x = x.split(':')

    x = x[1]

    return x

data['emergency_type'] = data['title'].apply(emergency_type_separator)

data.head()
call_types = data['call_type'].value_counts()

call_types
import datetime as dt

data['year'] = data['timeStamp'].dt.year

data['month'] = data['timeStamp'].dt.month_name()

data['day'] = data['timeStamp'].dt.day_name()

data['hour'] = data['timeStamp'].dt.hour
data.head()
font ={

    "size" :20

}

plt.figure(figsize=(7,6))

call_types.plot(kind="bar")

plt.xticks(rotation=0)

plt.xlabel("Types of Calls",fontdict=font)

plt.ylabel("No. of Calls",fontdict=font)

plt.title("Types of Calls Vs No. of Calls",fontdict=font)

plt.savefig("Types-of-Calls-vs-No-of-Calls.png")
calls_data = data.groupby(['month', 'call_type'])['call_type'].count()

calls_data
calls_percentage = calls_data.groupby(level=0).apply(lambda x:round(100*x/x.sum()) )

calls_percentage.head()
month_order =  ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

calls_percentage = calls_percentage.reindex(month_order, level=0)

calls_percentage.head()
plt.figure(figsize=(12,8))

calls_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls Per Month', fontdict=font)

plt.savefig("Calls-per-Month.png")
calls_data = data.groupby(["day","call_type"])["call_type"].count()

calls_data.head()
day_order= ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
calls_percentage = calls_data.groupby(level=0).apply(lambda x:round(100*x/x.sum()) )

calls_percentage.head()
calls_percentage = calls_percentage.reindex(day_order, level=0)

calls_percentage.head()
sns.set(rc={'figure.figsize':(12, 8)})

calls_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Day', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls Per Day', fontdict=font)

plt.savefig("Calls-per-Day.png")
calls_data = data.groupby(["hour","call_type"])["call_type"].count()

calls_data.head()
calls_percentage = calls_data.groupby(level=0).apply(lambda x:round(100*x/x.sum()) )

calls_percentage.head()
sns.set(rc={'figure.figsize':(12, 8)})

calls_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Hour', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls Per Hour', fontdict=font)

plt.savefig("Call-per-Hour.png")

ems_data = data.copy(deep=True)

ems_data.query('call_type == "EMS"',inplace = True)

ems_data.head()
count_ems = ems_data['emergency_type'].value_counts()
plt.figsize=(15,10)

plt.pie(count_ems.values[:7],labels=count_ems.index[:7],autopct="%.2f")

plt.savefig("Types-of-EMS-Calls-vs-No-of-Calls.png")
fire_data = data.copy(deep = True)

fire_data.query('call_type == "Fire"',inplace = True)

fire_data.head(2)
fire_data['emergency_type'].nunique()
count_fire = fire_data['emergency_type'].value_counts()

count_fire.head(2)
plt.figsize=(15,10)

plt.pie(count_fire.values[:7],labels=count_fire.index[:7],autopct="%.2f")

plt.savefig("Types-of-fire-Calls-vs-No-of-Calls.png")
traffic_data = data.copy(deep=True)
traffic_data.query('call_type == "Traffic"',inplace =True)

traffic_data.head(2)
traffic_data['emergency_type'].nunique()
count_traffic = traffic_data['emergency_type'].value_counts()

count_traffic.head(7)
plt.figsize=(15,10)

plt.pie(count_traffic.values[:5],labels=count_traffic[:5].index,autopct="%.2f")

plt.savefig("Types-of-traffic-Calls-vs-No-of-Calls.png")