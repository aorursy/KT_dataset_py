

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt #data visualization

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv(os.path.join(dirname, filename))

df.describe(), df.info()
df.head()
#Top 5 zip codes for 911 calls

#Use groupby

df.groupby(['zip']).count().sort_values(by=['lat'],ascending=False)['lat'].head(5)

#Use value_counts()

df['zip'].value_counts().head(5)
#Top 5 towns for 911 calls

df.groupby(['twp']).count().sort_values(by=['lat'],ascending=False)['lat'].head(5)

#Unique title codes

df['title'].nunique()
df['reason'] = df['title'].apply(lambda x: x.split(':')[0])

df['reason'].nunique()
df.groupby('reason').count()['lat']
df['reason'].value_counts().plot(kind = 'bar')

plt.xlabel('Reason')
type(df['timeStamp'].iloc[1])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])

time = df['timeStamp'].iloc[1]

time, time.hour, time.month,time.day,time.dayofweek
df['hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['month'] = df['timeStamp'].apply(lambda x: x.month)

df['day'] = df['timeStamp'].apply(lambda x: x.day)

df['dayofweek'] = df['timeStamp'].apply(lambda x: x.dayofweek)

daymap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['dayofweek'] = [daymap[i] for i in df['dayofweek']]

df['dayofweek']
import seaborn as sb
sb.countplot(x='dayofweek', hue ='reason',data=df)
sb.countplot(x='month', hue ='reason',data=df)
df_month = df.groupby(['month']).count()

df_month = df_month.reset_index()

sb.lmplot(x= 'month', y = 'lat', data = df_month)
#1 Use seaborn

df['date'] = df['timeStamp'].apply(lambda x: x.date())

sb.countplot(x ='date', data = df)

#2 Use plt

plt.figure(figsize=(20,7))

df_date = df.groupby(['date']).count()

df_date = df_date.reset_index()

plt.plot(df_date['date'],df_date['lat'])
#1.USe seaborn

plt.figure(figsize=(20,8))

sb.countplot(x ='date', hue ='reason',data = df)
#2. Use plt



reason = df['reason'].unique().tolist()



for i in reason:

    plt.figure(figsize=(20,5))

    df_reason= df[df['reason'] ==i]

    df_reason_bydate = df_reason.groupby(['date']).count()

    df_reason_bydate = df_reason_bydate.reset_index()

    plt.plot(df_reason_bydate['date'],df_reason_bydate['lat'])

    plt.title(i)
#Create a new dataset for the heatmap, using unstack

df_new = df.groupby(['dayofweek','hour']).count()['lat']

df_new = df_new.unstack(level = -1)

df_new.head()
sb.heatmap(df_new, cmap='viridis')
sb.clustermap(df_new,cmap='viridis')
df_monthday = df.groupby(['month','dayofweek']).count()['lat'].unstack(level = -1)

df_monthday
sb.heatmap(df_monthday)
sb.clustermap(df_monthday)