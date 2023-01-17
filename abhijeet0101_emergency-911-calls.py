import numpy as np

import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv("../input/911.csv")
df.isna().info()

df.info()
df.isna().sum()
df.head()
df.zip.value_counts().head()
df.twp.value_counts().tail()
df.title.nunique()
df["Reason"] = df.title.apply(lambda l : l.split(":")[0])
df["Reason"].head()
df.Reason.value_counts()
sns.countplot("Reason", data = df, palette ='viridis' )
type(df.timeStamp[0])
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
type(df['timeStamp'][0])
df['timeStamp'].head()
df['Hour'] = df.timeStamp.apply(lambda t : t.hour)

df['Month'] = df.timeStamp.apply(lambda t : t.month)

df['Day of Week'] = df.timeStamp.apply(lambda t : t.dayofweek)
df['Day of Week'].head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot('Day of Week', hue = 'Reason',data =df,palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1))
sns.countplot('Month', hue = 'Reason',data =df,palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1),loc = 2)
byMonth = df.groupby('Month').count()
byMonth
byMonth['twp'].plot()

plt.tight_layout()
df['Date'] = df.timeStamp.apply(lambda t: t.date())
df.groupby('Date').count()['twp'].plot()

plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
dayHour= df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')

dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis')