import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import datetime as dt

import matplotlib.dates as mdates

import numpy as np
air = pd.read_csv('../input/india-air-quality-data/data.csv',encoding='cp1252', low_memory=False)

air.sample(n=5)
air.shape
air.info()
air.describe()
sns.heatmap(air.isnull(),cbar=False,yticklabels=False,cmap = 'GnBu_r')
c = ['so2','no2','rspm','spm','pm2_5']

plt.figure(figsize=(10,5))

for i in range(0,len(c)):

    plt.subplot(5,1,i+1)

    sns.boxplot(air[c[i]],color='green',fliersize=5,orient='h')

    plt.tight_layout()

plt.figure(figsize=(6,5))

for i in range(0,len(c)):

    plt.subplot(5,1,i+1)

    sns.distplot(air[c[i]],kde=True)

    plt.tight_layout()

    
# sns.pairplot(air)
sns.heatmap(air.corr(),annot=True,cmap= 'Blues')
air['date'] = pd.to_datetime(air.date,format='%Y-%m-%d')

air.info()
air.drop(labels=['stn_code','sampling_date','pm2_5'],axis=1,inplace=True)
air.sample(n=2)
air['date'].isna().sum()
air['date'].fillna(method='ffill',inplace=True)
air.shape
air['date'].isnull().sum()
air['year'] = air['date'].dt.year
air.sample(2)
air['type'].value_counts().plot(kind='bar')
air['type'].replace("Sensitive Areas","Sensitive",inplace=True)

air['type'].replace("Sensitive Area","Sensitive",inplace=True)

air['type'].replace("Industrial Areas","Industrial",inplace=True)

air['type'].replace("Industrial Area","Industrial",inplace=True)

air['type'].replace("Residential and others","Residential",inplace=True)

air['type'].replace("RIRUO","Residential",inplace=True)
air['type'].value_counts().plot(kind='bar')
st_wise = air.pivot_table(values=['so2','no2','rspm','spm'],index='state').fillna(0)
maxso2 = st_wise.sort_values(by='so2',ascending=False)

maxso2.loc[:,['so2']].head(10).plot(kind='bar')
maxno2 = st_wise.sort_values(by='no2',ascending=False)

maxno2.loc[:,['no2']].head(10).plot(kind='bar')
maxrspm = st_wise.sort_values(by='rspm',ascending=False)

maxrspm.loc[:,['rspm']].head(10).plot(kind='bar')
maxspm = st_wise.sort_values(by='spm',ascending=False)

maxspm.loc[:,['spm']].head(10).plot(kind='bar')
kar_st = air.query('state=="Karnataka" ')
kar_st.head()
kar_st.type.value_counts().plot(kind='bar')
kar_st['spm'].mean()
kar_st['so2'].fillna(method='ffill',inplace=True)

kar_st['no2'].fillna(method='ffill',inplace=True)

kar_st['rspm'].fillna(method='ffill',inplace=True)

kar_st['spm'].fillna(168,inplace=True)
plt.figure(figsize=(18,6))

plt.xticks(np.arange(1987,2015))

mysore = kar_st.loc[ (kar_st['location']=='Mysore')]

sns.lineplot(x='year',y='so2',data=mysore)

sns.lineplot(x='year',y='no2',data=mysore)

plt.legend(['so2','no2'])
plt.figure(figsize=(18,6))

plt.xticks(np.arange(1987,2015))

mangalore = kar_st.loc[ (kar_st['location']=='Mangalore')]

sns.lineplot(x='year',y='so2',data=mangalore)

sns.lineplot(x='year',y='no2',data=mangalore)

plt.legend(['so2','no2'])
kar_st.tail()
air.agency.value_counts()
agent = air.loc[ (air['agency']=="Maharashtra State Pollution Control Board") ]
agent.type.value_counts().plot(kind='bar')
agent.year.value_counts().plot(kind='bar')
agent.location.value_counts().plot(kind='bar')
mah_loc = agent.loc[ (agent['location']=='Chandrapur') | (agent['location']=='Navi Mumbai') ] 

mah_loc.head()
mah_loc.so2.fillna(method='ffill',inplace=True)
mah_loc.groupby('location')['type'].value_counts().plot(kind='bar')
sns.barplot(x='type',y='so2',data=mah_loc)
sns.barplot(x='type',y='no2',data=mah_loc)