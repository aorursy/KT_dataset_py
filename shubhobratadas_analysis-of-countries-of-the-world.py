# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df=pd.read_csv('../input/countries-of-the-world/countries of the world.csv')
df
df.info()
df=df.dropna()








df.info()
df.head(30)
df.describe()
df.corr()
df['Region'].value_counts()
df=df.drop(['Other (%)'],axis=1)
df
df['Pop. Density (per sq. mi.)']=df['Pop. Density (per sq. mi.)'].astype('str')

df['Coastline (coast/area ratio)']=df['Coastline (coast/area ratio)'].astype('str')

df['Net migration']=df['Net migration'].astype('str')

df['Infant mortality (per 1000 births)']=df['Infant mortality (per 1000 births)'].astype('str')

df['GDP ($ per capita)']=df['GDP ($ per capita)'].astype('str')

df['Literacy (%)']=df['Literacy (%)'].astype('str')

df['Phones (per 1000)']=df['Phones (per 1000)'].astype('str')

df['Arable (%)']=df['Arable (%)'].astype('str')

df['Crops (%)']=df['Crops (%)'].astype('str')

df['Birthrate']=df['Birthrate'].astype('str')

df['Deathrate']=df['Deathrate'].astype('str')

df['Agriculture']=df['Agriculture'].astype('str')

df['Industry']=df['Industry'].astype('str')

df['Service']=df['Service'].astype('str')

df['Pop. Density (per sq. mi.)']=df['Pop. Density (per sq. mi.)'].str.replace(',','.')

df['Coastline (coast/area ratio)']=df['Coastline (coast/area ratio)'].str.replace(',','.')

df['Net migration']=df['Net migration'].str.replace(',','.')

df['Infant mortality (per 1000 births)']=df['Infant mortality (per 1000 births)'].str.replace(',','.')

df['GDP ($ per capita)']=df['GDP ($ per capita)'].str.replace(',','.')

df['Literacy (%)']=df['Literacy (%)'].str.replace(',','.')

df['Phones (per 1000)']=df['Phones (per 1000)'].str.replace(',','.')

df['Arable (%)']=df['Arable (%)'].str.replace(',','.')

df['Crops (%)']=df['Crops (%)'].str.replace(',','.')

df['Birthrate']=df['Birthrate'].str.replace(',','.')

df['Deathrate']=df['Deathrate'].str.replace(',','.')

df['Agriculture']=df['Agriculture'].str.replace(',','.')

df['Industry']=df['Industry'].str.replace(',','.')

df['Service']=df['Service'].str.replace(',','.')

df
df['Pop. Density (per sq. mi.)']=df['Pop. Density (per sq. mi.)'].astype('float')

df['Coastline (coast/area ratio)']=df['Coastline (coast/area ratio)'].astype('float')

df['Net migration']=df['Net migration'].astype('float')

df['Infant mortality (per 1000 births)']=df['Infant mortality (per 1000 births)'].astype('float')

df['GDP ($ per capita)']=df['GDP ($ per capita)'].astype('float')

df['Literacy (%)']=df['Literacy (%)'].astype('float')

df['Phones (per 1000)']=df['Phones (per 1000)'].astype('float')

df['Arable (%)']=df['Arable (%)'].astype('float')

df['Crops (%)']=df['Crops (%)'].astype('float')

df['Birthrate']=df['Birthrate'].astype('float')

df['Deathrate']=df['Deathrate'].astype('float')

df['Agriculture']=df['Agriculture'].astype('float')

df['Industry']=df['Industry'].astype('float')

df['Service']=df['Service'].astype('float')

df.info()
df['Climate'].value_counts()
df[df['Climate']=='2,5']


df=df.replace(to_replace ="2,5", 

                 value ="2") 
df=df.replace(to_replace ="1,5", 

                 value ="1") 


df['Climate'].value_counts()
df.head(30)
df['Climate']=df['Climate'].astype('int')
df.corr()
df.describe()
df[df['Pop. Density (per sq. mi.)']>2000]
df[df['Coastline (coast/area ratio)']>10]

df['Country'].nunique()
plt.figure(figsize=(40,10))

sns.countplot(df['Region'])

plt.show()


plt.figure(figsize=(40,10))

sns.boxplot(df['Population'])

plt.show()
plt.figure(figsize=(40,10))

sns.distplot(df['Pop. Density (per sq. mi.)'])

plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Pop. Density (per sq. mi.)'])

plt.show()
#Outliers cannot be removed as the data are valid.
plt.figure(figsize=(40,10))

sns.distplot(df['Coastline (coast/area ratio)'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Coastline (coast/area ratio)'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Net migration'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Infant mortality (per 1000 births)'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['GDP ($ per capita)'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Literacy (%)'])



plt.show()
plt.figure(figsize=(40,10))

sns.distplot(df['Literacy (%)'])



plt.show()
plt.figure(figsize=(40,10))

sns.distplot(df['Arable (%)'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Crops (%)'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Industry'])



plt.show()
plt.figure(figsize=(40,10))

sns.boxplot(df['Agriculture'])



plt.show()
plt.figure(figsize=(40,10))

sns.distplot(df['Agriculture'])



plt.show()
Rich=df['GDP ($ per capita)'].quantile(.75)

Rich
poor=df['GDP ($ per capita)'].quantile(.25)

poor
plt.figure(figsize=(30,10))

x=df[df['GDP ($ per capita)']>Rich]['Region']

sns.scatterplot(x, y=df['Country'])
plt.figure(figsize=(30,10))

x=df[df['GDP ($ per capita)']<poor]['Region']

sns.scatterplot(x, y=df['Country'])
#Conclusion



#1. Western Europe has most of the richest countries in the world 

#2. Sub-Saharan Africa has the most poor countries in the world
df['Literacy (%)'].mean()
more_literate=df[df['Literacy (%)']>df['Literacy (%)'].mean()][['Country','Literacy (%)','Industry']]

less_literate=df[df['Literacy (%)']<df['Literacy (%)'].mean()][['Country','Literacy (%)','Industry']]
more_literate
less_literate
plt.figure(figsize=(15,6))

sns.distplot(df[df['Literacy (%)']>df['Literacy (%)'].mean()]['Industry'])

sns.distplot(df[df['Literacy (%)']<df['Literacy (%)'].mean()]['Industry'])

#Conclusion 



#1. The countries with more literacy(%) has higher industry(%)
plt.figure(figsize=(15,6))

sns.distplot(df[df['Literacy (%)']>df['Literacy (%)'].mean()]['Agriculture'])

sns.distplot(df[df['Literacy (%)']<df['Literacy (%)'].mean()]['Agriculture'])

#conclusion 



#1. Countries which are little lagging behind in literacy(%) has more Agriculture
d=df.corr()
sns.heatmap(d)