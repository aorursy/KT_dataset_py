import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('../input/airquality/data.csv')
df.head()
df['type'].replace('Residential, Rural and other Areas','Residential',inplace = True)

df['type'].replace('Residential and others','Residential',inplace = True)

df['type'].replace('Industrial Areas','Industrial',inplace = True)

df['type'].replace('Industrial Area','Industrial',inplace = True)

df['type'].replace('Sensitive Area','Sensitive',inplace = True)

df['type'].replace('Sensitive Areas','Sensitive',inplace = True)
df['type'].value_counts()
df['type'].value_counts().plot(kind = 'bar')
g = df.groupby(['state','type'])

d = dict(list(g))

kar_ind = d[('Karnataka','Industrial')].median()

kar_res = d[('Karnataka','Residential')].median()

kar_sen = d[('Karnataka','Sensitive')].median()

print(kar_ind,kar_res,kar_sen)

# kar_riruo = d[('Karnataka','RIRUO')].mean()
print(df['so2'].isnull().sum())

print(df['no2'].isnull().sum())
df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Industrial') & (df['so2'].isnull()),'so2'] = kar_ind['so2']

df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Residential') & (df['so2'].isnull()),'so2'] = kar_res['so2']

df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Sensitive') & (df['so2'].isnull()),'so2'] = kar_sen['so2']

# df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'RIROU') & (df['so2'].isnull()),'so2'] = kar_rirou['so2']
df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Industrial') & (df['no2'].isnull()),'no2'] = kar_ind['no2']

df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Residential') & (df['no2'].isnull()),'no2'] = kar_res['no2']

df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'Sensitive') & (df['no2'].isnull()),'no2'] = kar_sen['no2']

# df.loc[(df['state'] == 'Karnataka') & (df['type'] == 'RIROU') & (df['so2'].isnull()),'so2'] = kar_rirou['so2']
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['year'].fillna(method = 'ffill',inplace = True)
df['year'] = df['year'].astype(int)
df['year'].isnull().sum()
d = dict(list(df[['location','year','so2','no2']].groupby('location')))
data = d['Bangalore'].groupby('year').median().reset_index()
data
plt.figure(figsize=(15,5))

plt.xticks(np.arange(1980,2016))

sns.lineplot(x='year',y='so2',data=data)

sns.lineplot(x='year',y='no2',data=data)

plt.legend(['so2','no2'])
df.head()
print(df.rspm.isnull().sum())

print(df.spm.isnull().sum())
df1 = dict(list(df.groupby(['location','type'])))

data = pd.DataFrame()

for key in df1:

    df2 = df1[key].sort_values('date')

    df2['rspm'].fillna(method = 'ffill',inplace = True)

    df2['spm'].fillna(method = 'ffill',inplace= True)

    data = pd.concat([data,df2])
df1 = dict(list(data.groupby(['location','type'])))

data1 = pd.DataFrame()

for key in df1:

    df2 = df1[key].sort_values('date')

    df2['rspm'].fillna(method = 'bfill',inplace = True)

    df2['spm'].fillna(method = 'bfill',inplace= True)

    data1 = pd.concat([data1,df2])
data1.head()
print(data1.rspm.isnull().sum())

print(data1.spm.isnull().sum())
df1 = dict(list(data1.groupby(['state','type'])))

data2 = pd.DataFrame()

for key in df1:

    df2 = df1[key]

    df2['rspm'].fillna(df2['rspm'].median(),inplace = True)

    df2['spm'].fillna(df2['spm'].median(),inplace= True)

    data2 = pd.concat([data2,df2])
print(data2.rspm.isnull().sum())

print(data2.spm.isnull().sum())
data2
df1 = dict(list(data2.groupby('type')))

data3 = pd.DataFrame()

for key in df1:

    df2 = df1[key]

    df2['rspm'].fillna(df2['rspm'].median(),inplace = True)

    df2['spm'].fillna(df2['spm'].median(),inplace= True)

    data3 = pd.concat([data3,df2])
data3
print(data3.rspm.isnull().sum())

print(data3.spm.isnull().sum())
data3['type'].value_counts()
data3.reset_index(inplace=True)
data3.drop(columns=['index','stn_code','sampling_date','agency','location_monitoring_station'],inplace = True)
data3.head()
data3.groupby('state').median()['rspm'].sort_values(ascending = False).plot(kind = 'bar', figsize = (17,5))
data3.groupby('location').median()['rspm'].sort_values(ascending = False).head(50).plot(kind = 'bar', figsize = (17,5))
data3.groupby('location').median()['rspm'].sort_values(ascending = False).tail(50).plot(kind = 'bar', figsize = (17,5))
data3.groupby('state').median()['spm'].sort_values(ascending = False).plot(kind = 'bar', figsize = (17,5))
data3.groupby('location').median()['spm'].sort_values(ascending = False).head(50).plot(kind = 'bar', figsize = (17,5))