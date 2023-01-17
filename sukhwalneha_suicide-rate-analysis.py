import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
suicide_df = pd.read_csv('../input/master.csv')
suicide_df.head()
suicide_df
suicide_df.info()
suicide_df.describe()
suicide_df.isnull().any()
sns.catplot('country','population',hue='age',data=suicide_df)
suicide_df = suicide_df.drop(['HDI for year','country-year'],axis=1)
suicide_df
min_year = min(suicide_df.year)

max_year = max(suicide_df.year)

print('Max year :',max_year)

print('Min year :',min_year)
#df = suicide_df.groupby(['country']).sum()
df = suicide_df[['country', 'suicides_no']]
df
df1 = df.groupby('country', as_index=False).sum()
df1
plt.figure(figsize=(15,20))

sns.barplot(x='suicides_no',y='country',data=df1)

df2 = suicide_df[['country', 'suicides_no','sex']]
df2.head()
plt.figure(figsize=(15,30))

sns.barplot(x='suicides_no',y='country',hue='sex',data=df2)
suicide_df.groupby('sex')['suicides_no'].sum().plot(kind='bar')
suicide_df.groupby('age')['suicides_no'].sum().plot(kind='bar')
suicide_df.groupby('year')['suicides_no'].sum().plot(kind='bar',figsize=(10,10))
pop = suicide_df[['country','population','suicides_no']]
pop.head()
pop1=pop.groupby('country',as_index=False).sum()
plt.figure(figsize=(50,15))

sns.barplot(x='population',y='suicides_no',data=pop1)
pop1.sort_values(by=['population'], inplace=True)
pop1.head()
pop1
df1.sort_values(by='suicides_no',inplace= True,ascending= False)
plt.figure(figsize=(15,20))

sns.barplot(x='suicides_no',y='country',data=df1)
plt.figure(figsize=(10,6))

sns.countplot(x='generation', hue='sex',data= suicide_df)
suicide_df.plot(x='generation',y='suicides_no',linestyle='',marker='o')