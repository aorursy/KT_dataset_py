import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
crime_df = pd.read_csv('../input/crimes-in-boston/crime.csv', encoding='latin-1')

oc_df = pd.read_csv('../input/crimes-in-boston/offense_codes.csv', encoding='latin-1')

oc_df = oc_df.rename(columns={'CODE': 'OFFENSE_CODE', 'NAME': 'OFFENSE_NAME'})

df = pd.merge(crime_df, oc_df, on='OFFENSE_CODE')

df.head()
df.shape
df.isnull().sum()
df['SHOOTING'].unique()

df['SHOOTING'] = df['SHOOTING'].apply(lambda x: 1 if x=='Y' else 0)

_ = sns.countplot(df['SHOOTING'])
df.isnull().sum()
df = df.dropna()

df.isnull().sum()
df.head()
plt.figure(figsize=(20, 10))

p = sns.countplot(df['OFFENSE_CODE_GROUP'])

plt.title('Offense Code Group')

_ = plt.setp(p.get_xticklabels(), rotation=90)
df_year = df.groupby(['YEAR']).size().reset_index(name='counts')

df_month = df.groupby(['MONTH']).size().reset_index(name='counts')
fig, axs = plt.subplots(2,2)

fig.set_figheight(15)

fig.set_figwidth(15)



p = sns.countplot(df['DAY_OF_WEEK'], ax=axs[0, 0])

q = sns.lineplot(x=df_month['MONTH'], y=df_month['counts'], ax=axs[1, 0], color='r')

r = sns.lineplot(x=df_year['YEAR'], y=df_year['counts'], ax=axs[0,1], color='g')

s = sns.countplot(df['DISTRICT'], ax=axs[1,1])
df_hour = df.groupby(['HOUR']).size().reset_index(name='counts')

fig, axs = plt.subplots(1,2)

fig.set_figheight(5)

fig.set_figwidth(15)



p = sns.countplot(df['HOUR'], ax=axs[0])

q = sns.lineplot(x=df_hour['HOUR'], y=df_hour['counts'], ax=axs[1], color='y')
df_date = df.groupby(['OCCURRED_ON_DATE']).size().reset_index(name='counts')

df_date['date'] =df_date.apply(lambda x: pd.to_datetime(x['OCCURRED_ON_DATE'].split(' ')[0]), axis=1)





plt.figure(figsize=(20, 10))

p = sns.lineplot(x=df_date['date'], y=df_date['counts'], color='r')
df.Lat.replace(-1, None, inplace=True)

df.Long.replace(-1, None, inplace=True)



plt.figure(figsize=(10, 10))

p = sns.scatterplot(x='Lat', y='Long', hue='DISTRICT',alpha=0.01, data=df)