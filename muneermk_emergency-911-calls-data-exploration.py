import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

sns.set(context = 'paper', style= "whitegrid", font_scale=2)
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

cf.go_offline()
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head()
df['twp'].value_counts().head()
plt.figure(figsize=(14,8))
df['twp'].value_counts().head(10).plot.bar(color = 'blue')
plt.xlabel('Townships', labelpad = 20)
plt.ylabel('Number of Calls')
plt.title('Townships with Most 911 Calls')
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])
df['Reason'].value_counts().head()
plt.figure(figsize=(14,8))
sns.countplot('Reason', data=df, palette='rainbow')
type(df['timeStamp'][0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

# Starting the hour value from 1 instead of 0
df['Hour'] = df['Hour'].map({0:1, 1:2, 2:3, 3:4, 4:5, 5:6, 6:7, 7:8, 8:9, 9:10, 10:11, 11:12, 12:13, 13:14, 
        14:15, 15:16, 16:17, 17:18, 18:19, 19:20, 20:21, 21:22, 22:23, 23:24})

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)

# Mapping the actual string names to the day of the week
df['Day of Week'] = df['Day of Week'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',
                                        4:'Fri',5:'Sat',6:'Sun'}) 

df['Year'] = df['timeStamp'].apply(lambda time: time.year)

plt.figure(figsize=(14,8))
sns.countplot(df['Day of Week'], data = df, hue = df['Reason'], palette='viridis')
plt.title('Count of the calls')
plt.legend(loc = 'center right', bbox_to_anchor=(1.2,0.5) )
plt.figure(figsize=(14,8))
sns.countplot(df['Month'], data = df, hue = df['Reason'], palette='viridis')
plt.title('Count of the calls in months')
plt.legend(loc = 'center right', bbox_to_anchor=(1.2,0.5) )
byMonth = df.groupby('Month').count()
byMonth['twp'].iplot(title =" Calls per month", xTitle='Month', yTitle='Calls')
df['Date'] = df['timeStamp'].apply(lambda x: x.date() )
df.head()
df.groupby('Date').count()['twp'].iplot(title =" Calls", xTitle='Month', yTitle='Calls')

df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].iplot(title ="Traffic", xTitle='Month', yTitle='Calls')

df[df['Reason']=='EMS'].groupby('Date').count()['twp'].iplot(title ="EMS", xTitle='Month', yTitle='Calls')
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].iplot(title ="Fire", xTitle='Month', yTitle='Calls')
df.groupby('Hour').count()['twp'].iplot(title ='Call by hour - All year', xTitle='Hour', yTitle='Calls')
df[df['Year']==2015].groupby('Hour').count()['twp'].iplot(title ='Call by hour - 2015', xTitle='Hour', yTitle='Calls')
df[df['Year']==2016].groupby('Hour').count()['twp'].iplot(title ='Call by hour - 2016', xTitle='Hour', yTitle='Calls')
df[df['Year']==2017].groupby('Hour').count()['twp'].iplot(title ='Call by hour - 2017', xTitle='Hour', yTitle='Calls')
df[df['Year']==2018].groupby('Hour').count()['twp'].iplot(title ='Call by hour - 2018', xTitle='Hour', yTitle='Calls')
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head()
plt.figure(figsize=(15,10))
sns.heatmap(dayHour, cmap = 'viridis', linewidths=.1)

plt.figure(figsize=(14,8))
sns.clustermap(dayHour, cmap = 'viridis', linewidths=.1)

dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
plt.figure(figsize=(14,8))
sns.heatmap(dayMonth, cmap = 'viridis', linewidths=.1)

plt.figure(figsize=(14,8))
sns.clustermap(dayMonth, cmap = 'viridis', linewidths=.1)

