import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from datetime import datetime
%matplotlib inline
df=pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['reason'] = df['title'].apply(lambda t: t.split(':')[0])
df['reason'].value_counts()
sns.countplot(x='reason', data=df, palette='viridis')
type(df['timeStamp'][0])
df['timeStamp'] = df['timeStamp'].apply(pd.to_datetime)
df['hour'] = df['timeStamp'].apply(lambda dt: dt.hour)

df['dayOfWeek'] = df['timeStamp'].apply(lambda dt: dt.strftime('%a'))

df['month'] = df['timeStamp'].apply(lambda dt: dt.month)
df.head()
## Display in spreadsheet format

df.groupby(['dayOfWeek','reason']).count()['e']
## Display in Pivot Table format

df.pivot_table(index='dayOfWeek', columns='reason', values='e', aggfunc='count')
## Visualised Display

sns.countplot(x='dayOfWeek', hue='reason', data=df, palette='viridis')

plt.legend(bbox_to_anchor=(1.02,0.72), loc=3)

plt.xlabel('day')
sns.countplot(x='month', data=df, hue='reason', palette='viridis')

plt.legend(bbox_to_anchor=(1.02,0.72), loc=3)

plt.xlabel('Month')
byMonth = df.groupby('month').count()

byMonth
byMonth['e'].plot()
byMonthReason = df.pivot_table(index='month', columns='reason', aggfunc='count')['e']
byMonthReason.plot()
sns.lmplot(x='month', y='twp', data=byMonth.reset_index())
df['date'] = df['timeStamp'].apply(datetime.date)
datetime.date(df['timeStamp'][0])
df.groupby('date')['e'].count().plot()

plt.tight_layout()

plt.xlabel('Date')
df[df['reason'] == 'Traffic'].groupby('date')['e'].count().plot()

plt.xlabel('Date')

plt.title('Traffic')

plt.tight_layout()
df[df['reason'] == 'Fire'].groupby('date')['e'].count().plot()

plt.title('Fire')

plt.tight_layout()
df[df['reason'] == 'EMS'].groupby('date')['e'].count().plot()

plt.title('EMS')

plt.tight_layout()
dayHour = df.pivot_table(index='dayOfWeek', columns='hour', values='e', aggfunc='count')



## Or Using 'groupby + unstack':

## dayHour = df.groupby(by=['dayOfWeek','hour']).count()['e'].unstack(level=-1)



## Reorder index by weekdays

dayList = 'Mon Tue Wed Thu Fri Sat Sun'.split()

dayHour.index = pd.Categorical(dayHour.index, dayList, ordered=True)

dayHour.index.name = 'dayOfWeek'



dayHour.sort_index(inplace=True)
sns.heatmap(data=dayHour, cmap='viridis')
sns.clustermap(data=dayHour, cmap='viridis')
dayMonth = df.groupby(['dayOfWeek','month']).count()['e'].unstack(level=1)

dayMonth.index = pd.Categorical(dayMonth.index, categories=dayList, ordered=True)

dayMonth.index.name = 'dayOfWeek'



## Reorder dayMonth index by weekdays

dayMonth.sort_index(inplace=True)

dayMonth
sns.heatmap(dayMonth, cmap='viridis')
sns.clustermap(dayMonth, cmap='viridis')