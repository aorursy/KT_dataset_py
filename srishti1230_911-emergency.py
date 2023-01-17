import pandas as pd
data=pd.read_csv('911.csv')
data.head()
data.shape
data.columns
data.title
def call_type(x):

    x=x.split(':')

    return x[0]
data['call_type']=data['title'].apply(call_type)
data['call_type'].unique()
call_types=data['call_type'].value_counts()

call_types
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
plt.figure(figsize=(12,6))

call_types.plot(kind='bar')

plt.xticks(rotation=30)

font={

    'size':20,

     'weight': 'bold'

}

plt.title("Total_no_of_different_calltypes",fontdict=font)

plt.xlabel("Call_types",fontdict=font)

plt.ylabel("no_of_calls",fontdict=font)
data.timeStamp.describe()
data['timeStamp']=pd.to_datetime(data['timeStamp'],infer_datetime_format=True)
data['timeStamp'].head()
import datetime as dt
data['Year']=data['timeStamp'].dt.year
data['Month']=data['timeStamp'].dt.month_name()
data['Day']=data['timeStamp'].dt.day_name()
data['Hour']=data['timeStamp'].dt.hour
data.columns
def emergency_type(x):

    x=x.split(':')

    return x[1]
data['emergency_type']=data['title'].apply(emergency_type)
data['emergency_type'].nunique()
data['emergency_type'].value_counts()
call_data=data.groupby(['Month','call_type'])['call_type'].count()
call_data
def percentage(x):

    p=round((100*x)/float(x.sum()))

    return p
call_data_percentage=call_data.groupby('Month').apply(percentage)
call_data_percentage
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
call_data_percentage = call_data_percentage.reindex(month_order,level=0)
call_data_percentage
call_data_percentage=call_data_percentage.reindex(['EMS','Traffic','Fire'],level=1)
call_data_percentage.head()
font={

    'size': 'x-large',

    'weight': 'bold'

}
plt.figure(figsize=(20, 5))

call_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Month', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Month', fontdict=font)
call_data=data.groupby(['Day','call_type'])['call_type'].count()
call_data
def percentage(x):

    p=round(x*100/float(x.sum()))

    return p
call_data_percentage=call_data.groupby('Day').apply(percentage)
call_data_percentage
call_data_percentage=call_data_percentage.reindex(['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'],level=0)
call_data_percentage=call_data_percentage.reindex(['EMS','Traffic','Fire'],level=1)

call_data_percentage
plt.figure(figsize=(20,5))

call_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the Day', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Day', fontdict=font)
call_data=data.groupby(['Hour','call_type'])['call_type'].count()
call_data
def percentage(x):

    p=round(x*100/float(x.sum()))

    return p
call_data_percentage=call_data.groupby('Hour').apply(percentage)
call_data_percentage
call_data_percentage=call_data_percentage.reindex(['EMS','Traffic','Fire'],level=1)

call_data_percentage
plt.figure(figsize=(20,5))

call_data_percentage.unstack().plot(kind='bar')

plt.xlabel('Name of the hour', fontdict=font)

plt.ylabel('Percentage of Calls', fontdict=font)

plt.xticks(rotation=0)

plt.title('Calls/Hour', fontdict=font)