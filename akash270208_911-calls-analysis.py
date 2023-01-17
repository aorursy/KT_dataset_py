import pandas as pd
data=pd.read_csv('/kaggle/input/montcoalert/911.csv')
data.head(2)
call=(data['title'].str.split(':',n=1,expand=True)) 
data['call_type']=call[0]
data.head(2)
data['timeStamp']=pd.to_datetime(data['timeStamp'],infer_datetime_format=True)
import matplotlib.pyplot as plt

import seaborn as sns
data.dtypes
data['hour']=data['timeStamp'].apply(lambda x: x.hour)

data['month']=data['timeStamp'].apply( lambda x: x.month)

data['Day of Week']=data['timeStamp'].apply(lambda x:x.weekday_name)
data.head(2)
data['call_type'].value_counts()
data['call_type'].value_counts().plot(kind='bar')
hourly_data=data.groupby('hour')['call_type'].value_counts()
hourly_data
sns.set(rc={'figure.figsize':(10,5)})

hourly_data.unstack().plot(kind='bar')
hourly_data=hourly_data.groupby(level=0).apply( lambda x:round(100*x/x.sum()))

hourly_data
hourly_data.unstack().plot(kind='bar')
hourly_data.unstack().plot(kind='bar', stacked=True)
daywise_data=data.groupby('Day of Week')['call_type'].value_counts()
daywise_data
day_order=[ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daywise_data=daywise_data.groupby(level=0).apply( lambda x:round(100*x/x.sum()))

daywise_data=daywise_data.reindex(day_order, level=0)
daywise_data.unstack().plot(kind='bar', stacked=True)