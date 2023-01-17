import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_csv('../input/parking-citations.csv')
#Issue Date to datetime
df['Issue Date'] = df['Issue Date'].apply(lambda x: str(x).split('T')[0])
df['Issue Date'] = pd.to_datetime(df['Issue Date'], infer_datetime_format=True)
df.set_index(df["Issue Date"],inplace=True)
df.head()
#Monthly Number of Incidents
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sbn.lineplot(data=df['Ticket number'].resample('M').count().truncate(before='2014'), ax=ax)
ax.set(title='Monthly Number of Incidents', xlabel='Time', ylabel='NO. of Incidents')
plt.show()
plt.rcParams['figure.figsize'] = 15,5
#group by Location
df_group = (df.groupby('Location', as_index=True)).agg({'Ticket number':'count'}).rename(columns={'Ticket number': 'Incidents Size'})
#select top 10 location based on incidents
df_group = df_group.sort_values(by='Incidents Size', ascending=False).iloc[0:10, :]
df_group.head(10)
#Number of Incidents of top 10 locations
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sbn.barplot(x=df_group.index, y=df_group['Incidents Size'], ax=ax)
ax.set(title='Incidents by Location', xlabel='Location', ylabel='NO. of Incidents')
plt.show()
plt.rcParams['figure.figsize'] = 28,5
#group by Violation code
df_group = (df.groupby('Violation code', as_index=True)).agg({'Ticket number':'count'}).rename(columns={'Ticket number': 'Incidents Size'})
#select top 10 location based on incidents
df_group = df_group.sort_values(by='Incidents Size', ascending=False).iloc[0:10, :]
df_group.head(10)
#Top 10 violation codes
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sbn.barplot(x=df_group.index, y=df_group['Incidents Size'], ax=ax)
ax.set(title='Incidents by Violation Code', xlabel='Violation Code', ylabel='NO. of Incidents')
plt.show()
plt.rcParams['figure.figsize'] = 23,8
#Monthly Amount Collected
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sbn.lineplot(data=df['Fine amount'].resample('M').sum().truncate(before='2014'), ax=ax)
ax.set(title='Monthly Fine Amount', xlabel='Time', ylabel='Fine Amount')
plt.show()
plt.rcParams['figure.figsize'] = 15,5