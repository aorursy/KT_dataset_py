import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import datetime as dt
%matplotlib inline 
df = pd.read_csv('../input/My Uber Drives - 2016.csv')
df.info()
df.head()
df.tail()
df1 = df.drop(df.index[1155])
df1['PICK_DATE'] = df['START_DATE*'].str.split(' ').str[0]
df1['DROP_DATE'] = df['END_DATE*'].str.split(' ').str[0]
test = [df1]

for dataset in test: 
    dataset['START_DATE*'] = pd.to_datetime(dataset['START_DATE*']).astype('datetime64[ns]')
    dataset['END_DATE*'] = pd.to_datetime(dataset['END_DATE*']).astype('datetime64[ns]')
df1['CITY_PAIR'] = df1['START*']+'-'+ df1['STOP*']

df1['TOTAL_TIME'] = df1['END_DATE*']-df1['START_DATE*']
df1.info()

df1.isnull().sum()
df1['PURPOSE*'] = df1['PURPOSE*'].fillna('OTHER')
df1.groupby('PURPOSE*', as_index=False).sum()
df1.isnull().sum()
data = [df1]

for dataset in data:
    dataset['CATEGORY*'][df1['PURPOSE*']=='Meal/Entertain'] = 'Meals'
df1.describe()
df1.describe(include=['O'])
oth = ['OTHER']

g = sns.FacetGrid(data=df1[~df1['PURPOSE*'].isin(oth)], aspect=2, size=6)
g.map(sns.boxplot, 'PURPOSE*', 'MILES*', palette="Set1")
plt.show()
df1.head()
plt.figure(figsize=(10,10))
df1['PURPOSE*'].value_counts()[:11].plot(kind='pie',autopct='%1.1f%%',shadow=True,explode=[0.1,0,0,0,0,0,0,0,0,0,0])
plt.show()

g = sns.FacetGrid(data=df1, aspect=2, size=8)
g.map(sns.countplot, 'PURPOSE*', palette="Set1")
plt.show()

plt.figure(figsize=(18,8))
df1['CITY_PAIR'].value_counts()[:50].plot(kind='bar')
plt.show()