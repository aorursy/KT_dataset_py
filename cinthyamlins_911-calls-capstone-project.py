import pandas as pd

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/911.csv')
df.info()
df.head()
df['zip'].value_counts().head()
df['twp'].value_counts().head()
df['title'].nunique()
df['Reason'] = df['title'].apply(lambda x: x.split(':')[0])

df.head(1)
df['Reason'].value_counts().head(1)

sns.set_style('whitegrid')
sns.countplot(data=df, x='Reason');
# df['timeStamp'].dtypes



type(df['timeStamp'].iloc[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda x: x.hour)

df['Month'] = df['timeStamp'].apply(lambda x: x.month)

df['Day of Week'] = df['timeStamp'].apply(lambda x: x.dayofweek)

df.head(1)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)

df.head(1)
sns.countplot(data=df, x='Day of Week', hue='Reason', palette='rainbow')

plt.legend(loc='best', bbox_to_anchor=(1, 0.7));


# mmap = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

# df['Month'] = df['Month'].map(mmap)

# months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
sns.countplot(data=df, x='Month', hue='Reason', palette='viridis')

plt.legend(loc='best', bbox_to_anchor=(1, 0.7));
byMonth = df.groupby('Month').count()

# byMonth.apply(months)

byMonth.head(2)
byMonth['twp'].plot();
sns.lmplot(data=byMonth.reset_index(), x='Month', y='twp');
df['Date'] = df['timeStamp'].apply(lambda x: x.date());

df.head(2)
byDate = df.groupby('Date').count()

byDate['twp'].plot();

plt.tight_layout()

df['Reason'].unique()
plt.figure(figsize=(18,6));

df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot();

plt.legend(loc='best', bbox_to_anchor=(1, 1));
plt.figure(figsize=(18,6))

df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot();

plt.legend(loc='best', bbox_to_anchor=(1, 1));
plt.figure(figsize=(15,5))

df[df['Reason']=='Traffic'].groupby("Date").count()['twp'].plot()

plt.legend(loc='best', bbox_to_anchor=(1,1));
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
sns.heatmap(dayHour);
sns.clustermap(dayHour);
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth.head()
sns.heatmap(dayMonth);
sns.clustermap(dayMonth, cmap='viridis');
g = sns.PairGrid(dayMonth)

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot);
sns.pairplot(dayMonth);