import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../input/2018_03.csv")
for i in range(4,10):
    df = df.append(pd.read_csv("../input/2018_0"+str(i)+".csv"))
df.head()
df = df[df['type'] == 'NJ Transit']
df.head()
df.isnull().sum()
df[df.isnull().any(axis=1)].head()
df = df.dropna()
df.describe()
df.describe(exclude=[np.number])
sns.distplot(df['delay_minutes'])
sns.countplot(df['status'])
df[df['status'] == 'cancelled'].head()
df['long_delay'] = df['delay_minutes'] > 5
df.groupby('line')['long_delay'].mean().sort_values(ascending=False).plot(kind='bar')
x = df.groupby(['line', 'status']).size().unstack()
x['cancelled']/(x['departed']+x['estimated'])
x['cancelled']
ax = df.groupby('stop_sequence')["delay_minutes"].mean().plot()
ax.set_ylabel("average delay_minutes")
df.groupby('from')['delay_minutes'].mean().sort_values(ascending=False).head(10)
df.groupby('to')['delay_minutes'].mean().sort_values(ascending=False).head(10)
df.date = pd.to_datetime(df.date)
x = df.groupby('date')['delay_minutes'].mean()
fig, ax = plt.subplots()
fig.set_size_inches(20,8)
fig.autofmt_xdate()
ax.plot(x)
ax.set_ylabel('average delay_minutes')
plt.show()
df.scheduled_time = pd.to_datetime(df.scheduled_time)
df['time'] = df.scheduled_time.dt.time

fig, ax = plt.subplots()
fig.set_size_inches(20,8)
x = df.groupby('time')['delay_minutes'].mean()
ax.plot(x)
ax.set_ylabel('average delay_minutes')
plt.show()
x.sort_values(ascending=False).head(5)
x = df.groupby(['date', 'status']).size().unstack()
x = (x['cancelled']/(x['departed']+x['estimated']))

fig, ax = plt.subplots()
fig.set_size_inches(20,8)
ax.plot(x)
ax.set_ylabel('cancellation rate')
plt.show()
x.sort_values(ascending=False).head(5)
x = df.groupby(['time', 'status']).size().unstack()
x = (x['cancelled']/(x['departed']+x['estimated']))

fig, ax = plt.subplots()
fig.set_size_inches(20,8)
ax.plot(x)
ax.set_ylabel('cancellation rate')
plt.show()
