# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings('ignore')



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/master.csv')
df.head()
df.describe()
df.info()
df.dtypes
#lets drop all those columns which contains more the 30% null value

#we can drop HDI and country-year columns

df2 = df[[col for col in df if df[col].count()/len(df)>=.50]]
df2.drop('country-year',axis=1,inplace=True)
df2.head()
#Now lets change the catagorical datato numeric category

age = df['age'].unique()

A ={}

j=0

for i in age:

    A[i] = j

    j+=1

df2['age'] = df2['age'].map(A)

a = {v: k for k, v in A.items()}

df2.head()
gen = df['generation'].unique()

G ={}

j=0

for i in gen:

    G[i] = j

    j+=1

df2['generation'] = df2['generation'].map(G)

g = {v: k for k, v in G.items()}

df2.head()
sex = df['sex'].unique()

S ={}

j=0

for i in sex:

    S[i] = j

    j+=1

df2['sex'] = df['sex'].map(S)

S

s = {v: k for k, v in S.items()}

df2.head()
country = df['country'].unique()

C ={}

j=0

for i in country:

    C[i] = j

    j+=1

df2['country'] = df['country'].map(C)

c = {v: k for k, v in C.items()}

df2.head()
df2.tail()
df2.corr()
sns.heatmap(df2.corr());
county = df[['country','suicides_no']].groupby('country',as_index=False).sum().sort_values(by='suicides_no',ascending=False)

fig=plt.figure(figsize=(20,10))

sns.barplot(x=county['country'],y=county['suicides_no'],data=county)

plt.xticks(rotation=90)

plt.title('World-wide total suicides from 1985-2016');
country = df[['country','suicides_no']].groupby('country',as_index=False).sum().sort_values(by='suicides_no',ascending=False).head(10)

sns.barplot(x='country',y='suicides_no',data=country)

plt.xticks(rotation=90)

plt.title('Top 10 countries in total suicides');
age_suicide = df[['age','sex','suicides_no']].groupby(['age','sex'],as_index=False).sum()

sns.barplot(x='age',y='suicides_no',hue='sex',data=age_suicide)

plt.xticks(rotation=90)

plt.title('Age wise total suicides');
sns.barplot(x='generation',y='suicides_no',hue='sex',data=df[['generation','suicides_no','sex']].groupby(['generation','sex'],as_index=False).sum())

plt.xticks(rotation=90)

plt.title('Generation wise total suicides');
a = df[['sex','suicides_no']].groupby('sex',as_index=False).sum()

sns.barplot(x='sex',y='suicides_no',data=a)

plt.title('Gender wise total suicides');
sns.barplot(x='year',y='suicides_no',data=df[['year','suicides_no']].groupby('year',as_index=False).sum())

plt.xticks(rotation=90)

plt.title('Year wise total suicides');
data=df[['year','suicides_no','sex','country']].groupby(['year','country','sex'],as_index=False).sum().sort_values(by=['suicides_no'],ascending=False)

data[data['year']==2015].head(10)
data=df[['year','suicides_no','sex','country']].groupby(['year','country','sex'],as_index=False).sum().sort_values(by=['suicides_no'],ascending=False)

data[data['year']==2016].head(15)