import pandas as pd
data=pd.read_csv('../input/911.csv')
data.head()
data.title.head()
def title_sep(x):

    x=x.split(':')

    return x[0]
data['type_of_call']=data['title'].apply(title_sep)
data.head()
data['type_of_call'].unique()
call_type=data['type_of_call'].value_counts()
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
font={

    'size':20

}
from decimal import Decimal
plt.figure(figsize=(10,5))

px=call_type.plot(kind='bar')

for p in px.patches:

    px.annotate(Decimal(str(p.get_height())), (p.get_x(), p.get_height()))

plt.xticks(rotation=30)

plt.savefig('type_of_call.png')
data.info()
data['timeStamp']=pd.to_datetime(data['timeStamp'], infer_datetime_format=True)
data.info()
import datetime as dt
data['Year']=data['timeStamp'].dt.year
data['Month']=data['timeStamp'].dt.month_name()
data['Day']=data['timeStamp'].dt.day_name()
data['Hour']=data['timeStamp'].dt.hour
data.head()
calls_month = data.groupby(['Month', 'type_of_call'])['type_of_call'].count()
calls_month
calls_month_percentage = calls_month.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))
calls_month_percentage
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
calls_month_percentage = calls_month_percentage.reindex(month_order, level=0)
calls_month_percentage
calls_month_percentage = calls_month_percentage.reindex(['EMS','Traffic','Fire'], level=1)
calls_month_percentage
sns.set(rc={'figure.figsize':(12, 8)})

calls_month_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Month', fontdict=font)

plt.savefig('call_vs_month.png')
calls_hour = data.groupby(['Hour', 'type_of_call'])['type_of_call'].count()
calls_hour
calls_hour_percentage = calls_hour.groupby(level=0).apply(lambda x:round(100*x/float(x.sum())))
calls_hour_percentage
calls_hour_percentage = calls_hour_percentage.reindex(['EMS','Traffic','Fire'], level=1)
calls_hour_percentage
sns.set(rc={'figure.figsize':(12, 8)})

calls_hour_percentage.unstack().plot(kind='bar')

plt.xlabel('Hour of the day', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Month', fontdict=font)

plt.savefig('call-vs-hour.png')
data.head()
def spliter(x):

    x=x.split(':')

    return x[1]
data['emergency_call']=data['title'].apply(spliter)
data.head()
data['emergency_call'].unique()
emergency_call=data['emergency_call'].value_counts()
emergency_call
emergency_call_percentage = emergency_call.groupby(level=0).apply(lambda x:round(100*x/float(emergency_call.sum())))
emergency_call_percentage
emergency_call_percentage=emergency_call_percentage.head(38)
sns.set(rc={'figure.figsize':(20, 6)})

emergency_call_percentage.plot(kind='bar')

plt.xlabel('type_of_emergency_calls', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=90)

plt.title('Calls/types_of_emergency_calls', fontdict=font)

plt.savefig('call-vs-types_of_emergency.png')
data.head()
data1=data[data['type_of_call']=='EMS']
data1.head()
data1['emergency_call'].unique()
ems_data=data1['emergency_call'].value_counts().head()
data2=data[data['type_of_call']=='Traffic']
Traffic_data=data2['emergency_call'].value_counts().head()
data3=data[data['type_of_call']=='Fire']
Fire_data=data3['emergency_call'].value_counts().head()
plt.figure(figsize=(10,8))

plt.pie(ems_data.values,labels=ems_data.index,autopct="%.2f")

plt.savefig('EMS_top_5.png')
plt.figure(figsize=(10,8))

plt.pie(Traffic_data.values,labels=Traffic_data.index,autopct="%.2f")

plt.savefig('Traffic_top_5.png')
plt.figure(figsize=(10,8))

plt.pie(Fire_data.values,labels=Fire_data.index,autopct="%.2f")

plt.savefig('fire_top_5.png')