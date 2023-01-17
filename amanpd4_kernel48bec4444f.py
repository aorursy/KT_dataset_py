import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
df = pd.read_csv('../input/montcoalert/911.csv')
df.info()
df.head()
df['zip'].value_counts().head()
df['twp'].value_counts().head()
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda x:x[:].split(':')[0])

df['Reason']
df['Reason'].value_counts().head()
sns.countplot(x='Reason',data=df,palette='viridis')
type('timeStamp')
df.head(1)
df['timeStamp']=pd.to_datetime(df['timeStamp'])
df['hour']= df['timeStamp'].apply(lambda x:x.hour)

df['month'] = df['timeStamp'].apply(lambda month:month.month)

df['dayOfWeek']= df['timeStamp'].apply(lambda day:day.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['dayOfWeek'] = df['dayOfWeek'].map(dmap)
sns.countplot(x='dayOfWeek',data=df,hue='Reason',palette='viridis')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='month',data=df,hue='Reason',palette='viridis')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
bymonth = df.groupby('month').count()

bymonth.head()
bymonth['zip'].plot()
sns.lmplot(x='month',y='twp',data=bymonth.reset_index())
df.head(1)
df['Date']=df['timeStamp'].apply(lambda x:x.date())

df['Date']
bydate=df.groupby('Date').count()

bydate['twp'].plot()

plt.tight_layout()
df['Reason'].value_counts()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
dayHour = df.groupby(by=['dayOfWeek','hour']).count()['Reason'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
dayMonth = df.groupby(by=['dayOfWeek','month']).count()['Reason'].unstack()

dayMonth.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayMonth,cmap='viridis')