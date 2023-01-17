### 1) Q : max(Ems,Fire,Traffic)

### 2) Q : timestamp max(year),max(month),max(day),max(hours)

### 3) Q : Folium Library ((pie chart:ems,fire,traffic)100 data points)

### 4) Q : 
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import seaborn as sns
data = pd.read_csv("../input/911.csv")

data.head()
data.shape
data.columns
data.info()
data['timeStamp'] = pd.to_datetime(data['timeStamp'])

data.head(5)
import re

def type_reason(x):

    x = str(x)

    if (re.search("EMS", x)):

        return "EMS"

    

    elif (re.search("Traffic", x)):

        return "Traffic"

    else:

        return "Fire"
data["call_type"] = data["title"].apply(type_reason)

data.head()
datax_major_type=data["call_type"].value_counts()
datax_major_type = pd.Series(datax_major_type)

datax_major_type
font ={

    "size" :20

}

plt.figure(figsize=(8,6))

datax_major_type.plot(kind="bar")

plt.xticks(rotation=30)

plt.xlabel("Types of Calls",fontdict=font)

plt.ylabel("No. of Calls",fontdict=font)

plt.title("Types of Calls Vs No. of Calls",fontdict=font)

plt.savefig("Types-of-Calls-vs-No-of-Calls.png")
data['Year'] = data['timeStamp'].dt.year

data.head()
data['Month'] = data['timeStamp'].dt.month_name()

data.head()
data["Day"] = data['timeStamp'].dt.day_name()

data.head()
data["Hour"] = data['timeStamp'].dt.hour

data.head()
def actual_type_call(x):

    x= x.split(':')

    return x[1]
data["emergency_reason"] = data["title"].apply(actual_type_call)

data.head()
ems_data = data.copy(deep=True)
ems_data.query('call_type == "EMS"',inplace = True)

ems_data.head()
count_ems = ems_data['emergency_reason'].value_counts()
plt.figsize=(15,10)

plt.pie(count_ems.values[:7],labels=count_ems.index[:7],autopct="%.2f")

plt.savefig("Types-of-EMS-Calls-vs-No-of-Calls.png")
### Fire Data first 7 pie plot
fire_data = data.copy(deep = True)
fire_data.query('call_type == "Fire"',inplace = True)

fire_data.head(2)
fire_data['emergency_reason'].nunique()
count_fire = fire_data['emergency_reason'].value_counts()

count_fire.head(2)
plt.figsize=(15,10)

plt.pie(count_fire.values[:7],labels=count_fire.index[:7],autopct="%.2f")

plt.savefig("Types-of-fire-Calls-vs-No-of-Calls.png")
### Traffic Pie Chart
traffic_data = data.copy(deep=True)
traffic_data.query('call_type == "Traffic"',inplace =True)

traffic_data.head(2)
traffic_data['emergency_reason'].nunique()
count_traffic = traffic_data['emergency_reason'].value_counts()

count_traffic.head(7)
plt.figsize=(15,10)

plt.pie(count_traffic.values[:5],labels=count_traffic[:5].index,autopct="%.2f")

plt.savefig("Types-of-traffic-Calls-vs-No-of-Calls.png")
calls_data = data.groupby(["Month","call_type"])["call_type"].count()

calls_data.head()
calls_percentage = calls_data.groupby(level=0).apply(lambda x:round(100*x/x.sum()) )

calls_percentage.head()
month_order =  ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
calls_percentage = calls_percentage.reindex(month_order, level=0)

calls_percentage.head()
sns.set(rc={'figure.figsize':(12, 8)})

calls_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls Per Month', fontdict=font)

plt.savefig("Calls-per-Month.png")
calls_data = data.groupby(["Day","call_type"])["call_type"].count()

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
calls_data = data.groupby(["Hour","call_type"])["call_type"].count()

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