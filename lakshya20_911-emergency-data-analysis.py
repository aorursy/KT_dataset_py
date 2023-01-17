import pandas as pd
data = pd.read_csv('../input/911.csv')
data.head()
data.title.nunique()
def callseperator(x):

    x = x.split(':')

    return x[0]

def callpurpose(y):

    y = y.split(':')

    return y[1]
data['call_types'] = data['title'].apply(callseperator)
data['call_purpose'] = data['title'].apply(callpurpose)
data.head()
data['call_types'].nunique()
data['call_types'].value_counts()
call_types = data['call_types']
call_types
call_purpose = data['call_purpose']
call_purpose
data.drop('title',axis=1,inplace=True)
data.head()
data['timeStamp'] = pd.to_datetime(data['timeStamp'])
data.head()
data.info()
data['year'] = data['timeStamp'].dt.year
data['month'] = data['timeStamp'].dt.month_name()
data['day'] = data['timeStamp'].dt.day_name()
data['hour'] = data['timeStamp'].dt.hour
data.head()
data.info()
def stringconv(x):

    x = str(x)

    return x
data['year'] = data['year'].apply(stringconv)
data.info()
calls_year = data.groupby('year')['call_types']
calls_year.head()
calls_month = data.groupby('month')['call_types']
calls_month.head()
calls_year.value_counts()
calls_month = calls_month.value_counts()
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()
calls_month
plt.figure(figsize=(12, 8))

calls_month.unstack().plot(kind='bar')

plt.legend()
calls_days = data.groupby('day')['call_types']
calls_days = calls_days.value_counts()
calls_days
plt.figure(figsize=(12, 8))

calls_days.unstack().plot(kind='bar')

plt.legend()
calls_month_percentage = calls_month.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
calls_month_percentage
plt.figure(figsize=(12, 8))

calls_month_percentage.unstack().plot(kind='bar')

plt.legend()
calls_days_percentage = calls_days.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
plt.figure(figsize=(12, 8))

calls_days_percentage.unstack().plot(kind='bar')

plt.legend()
call_types = call_types.value_counts()
call_types
sns.set(rc=({'font.size':30}))

plt.figure(figsize=(12,12))

plt.pie(call_types,labels=call_types.index,autopct='%.2f')
plt.figure(figsize=(12, 8))

call_types.plot(kind='bar')

plt.legend()
data.head()
ems_data = data[data['call_types'] == 'EMS']
ems_data = ems_data['call_purpose'].value_counts()[:10]
ems_data_percent = round(100 * ems_data/ems_data.sum())
ems_data_percent
plt.figure(figsize=(12, 8))

ems_data.plot(kind='bar')

plt.legend()
sns.set(rc=({'font.size':30}))

plt.figure(figsize=(12,12))

plt.pie(ems_data,labels=ems_data.index,autopct='%.2f')
data.head(10)
plt.figure(figsize=(12, 8))

ems_data_percent.plot(kind='bar')

plt.legend()
fire_data = data[data['call_types'] == 'Fire']
fire_data = fire_data['call_purpose'].value_counts()[:10]
fire_data_percent = round(100 * fire_data/fire_data.sum())
fire_data_percent
plt.figure(figsize=(12, 8))

fire_data.plot(kind='bar')

plt.legend()
sns.set(rc=({'font.size':25}))

plt.figure(figsize=(12,12))

plt.pie(fire_data,labels=fire_data.index,autopct='%.2f')
plt.figure(figsize=(12, 8))

fire_data_percent.plot(kind='bar')

plt.legend()
traffic_data = data[data['call_types'] == 'Traffic']
traffic_data = traffic_data['call_purpose'].value_counts()[:10]
traffic_data_percent = round(100 * traffic_data/traffic_data.sum())
traffic_data_percent
plt.figure(figsize=(12, 8))

traffic_data.plot(kind='bar')

plt.legend()
sns.set(rc=({'font.size':20}))

plt.figure(figsize=(12,12))

plt.pie(traffic_data[:5],labels=traffic_data.index[:5],autopct='%.2f')
plt.figure(figsize=(12, 8))

traffic_data_percent.plot(kind='bar')

plt.legend()