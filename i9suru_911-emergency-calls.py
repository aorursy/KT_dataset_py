import pandas as pd

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

sns.set()
data=pd.read_csv('../input/911.csv')
data.head()
data.shape
data.info()
data.columns
data['title'].head()
data['call_type']=data['title'].apply(lambda x :x.split(':')[0])
data.head()
data['call_type'].nunique()
data['call_type'].unique()
call_types=data['call_type'].value_counts()
from decimal import Decimal

plt.figure(figsize=(15, 5))

ax = call_types.plot(kind='bar')

for p in ax.patches:

    ax.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))

plt.xticks(rotation=0)

plt.savefig("1.png")
data['timeStamp']=pd.to_datetime(data['timeStamp'])
data['timeStamp'].head()
data.info()
import datetime as dt
data['Year']=data['timeStamp'].dt.year
data['Month']=data['timeStamp'].dt.month_name()
data['Hour']=data['timeStamp'].dt.hour
data['day']=data['timeStamp'].dt.day_name()
data.head()
data['emergency_type'] = data['title'].apply(lambda x:x.split(':')[1])

data.head()
calls_data = data.groupby(['Month', 'call_type'])['call_type'].count()
calls_data_percentage = calls_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))

calls_data_percentage.head()

calls_data_percentage = calls_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)

month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

calls_data_percentage = calls_data_percentage.reindex(month_order, level=0)

calls_data_percentage.head()

font = {

    'size': 'x-large',

    'weight': 'bold'

}
sns.set(rc={'figure.figsize':(12, 8)})

calls_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Month', fontdict=font)

plt.savefig('2monthly.png')
hours_data = data.groupby(['Hour', 'call_type'])['call_type'].count()

hours_data.head()

hours_data_percentage = hours_data.groupby(level=0).apply(lambda x: round(100*x/float(x.sum())))
hours_data_percentage.head()
hours_data_percentage = hours_data_percentage.reindex(['EMS', 'Traffic', 'Fire'], level=1)

sns.set(rc={'figure.figsize':(18, 8)})

hours_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Hour of the day', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Hour', fontdict=font)

plt.savefig('2hourly.png')
ems_data=data[data['call_type']=='EMS']['emergency_type'].value_counts()[:5]
fire_data=data[data['call_type']=='Fire']['emergency_type'].value_counts()[:5]
traffic_data=data[data['call_type']=='Traffic']['emergency_type'].value_counts()[:5]
plt.pie(ems_data,labels=ems_data.index,autopct="%.2f")

plt.savefig("3.png")
plt.pie(fire_data,labels=fire_data.index,autopct="%.2f")

plt.savefig("4.png")
plt.pie(traffic_data,labels=traffic_data.index,autopct="%.2f")

plt.savefig("5.png")