import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

df=pd.read_csv('../input/montcoalert/911.csv')
df.head()
df.info()
df['zip'].value_counts().head(10)

by_zip = df.groupby(['zip']).count()

by_zip.sort_values(by='addr', ascending=False, inplace=True)

by_zip = by_zip.head(10)

plt.figure(figsize=(12,6))

plt.title('Top 10 zip codes on 911 calls:')

sns.barplot(x='zip', y='addr', data=by_zip.reset_index()) 

df['twp'].value_counts().head(10)
by_twp = df.groupby(['twp']).count()

by_twp.sort_values(by='addr', ascending = False, inplace=True)

by_twp = by_twp.head(10)

plt.figure(figsize=(20,6))

plt.title('Top 10 cities on 911 calls')

sns.barplot(x='twp', y='addr', data=by_twp.reset_index())
reasons = df['title'].apply(lambda reason: reason.split(':')[0])

df['reason'] = reasons

df.head()
df['reason'].value_counts()
plt.title("Most common reasons on 911 calls")

sns.countplot(x='reason', data=df) 

c_reasons = df['reason'].value_counts().to_frame() # Converting the series into a dataframe:

c_reasons.plot(kind='pie', subplots=True, figsize=(6,6), title="Most common reasons on 911 calls")
df['timeStamp'] = pd.to_datetime(df['timeStamp']) # Converting by the to_datetime()
df['timeStamp'].iloc[0].hour
df['timeStamp'].iloc[0].month
df['timeStamp'].iloc[0].dayofweek
df['timeStamp'].iloc[0].year
df['hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['dayofweek'] = df['timeStamp'].apply(lambda time: time.dayofweek)

df['month'] = df['timeStamp'].apply(lambda time: time.month)

df['year'] = df['timeStamp'].apply(lambda time: time.year)
df.head()
days = {0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
df['dayofweek'] = df['dayofweek'].map(days)
df['dayofweek']
df['dayofweek'].value_counts()
plt.title("911 calls by days of week:")

sns.countplot(x='dayofweek', data=df)
plt.title('911 Calls by day of week and reasons')

sns.countplot(x='dayofweek', data=df, hue='reason')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='month', data=df)

plt.title("911 calls by month")
months = {1:'Jan', 2: 'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

df_mn = df.copy() # Just making a copy to preserve temporal order on future insights

df_mn['month'] = df_mn['month'].map(months)
df_mn['month']
plt.title("911 calls by month")

sns.countplot(x='month', data=df_mn)
plt.title("911 calls by month")

sns.countplot(x='month', data=df_mn, hue='reason')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title("911 calls per year")

sns.countplot(x='year', data=df)
plt.title("911 calls per year")

sns.countplot(x='year', data=df, hue='reason')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
df_2015 = df[df['year'] == 2015]

df_2015['month'].value_counts()
by_month = df.groupby(['month']).count()
by_month.head()
by_month['addr'].plot(title="911 calls through months")
a_2016 = df[df['year']==2016].groupby(['month']).count()

a_2017 = df[df['year']==2017].groupby(['month']).count()

a_2018 = df[df['year']==2018].groupby(['month']).count()

a_2019 = df[df['year']==2019].groupby(['month']).count()

a_2016['addr'].plot(legend=True)

a_2017['addr'].plot()

a_2018['addr'].plot()

a_2019['addr'].plot()

plt.title("911 Calls through months year by year")

plt.legend(['2016','2017','2018','2019'])
sns.lmplot(x='month', y='addr', data=by_month.reset_index())
df[df['reason'] == 'Traffic'].groupby(['month']).count()['addr'].plot()

df[df['reason'] == 'EMS'].groupby(['month']).count()['addr'].plot()

df[df['reason'] == 'Fire'].groupby(['month']).count()['addr'].plot()

plt.title('Reasons of 911 calls line comparison')

plt.legend(['Traffic','EMS', 'Fire'])

plt.tight_layout()
day_hour = df.groupby(['dayofweek', 'hour']).count()['reason'].unstack()

day_hour.head()
plt.figure(figsize=(12,6))

plt.title("911 Calls on days of week and hour")

sns.heatmap(day_hour, cmap='YlGnBu')
sns.clustermap(day_hour, cmap='YlGnBu')
day_month = df.groupby(['dayofweek', 'month']).count()['reason'].unstack()

day_month.head()
plt.figure(figsize=(12,6))

sns.heatmap(day_month, cmap="inferno_r")
sns.clustermap(day_month, cmap='inferno_r')