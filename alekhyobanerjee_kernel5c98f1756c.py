# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')
data
data.columns
data['Pop. Density (per sq. mi.)']=data['Pop. Density (per sq. mi.)'].str.replace(',','.').astype('float')

data['Coastline (coast/area ratio)']=data['Coastline (coast/area ratio)'].str.replace(',','.').astype('float')

data['Net migration']=data['Net migration'].str.replace(',','.').astype('float')

data['Infant mortality (per 1000 births)']=data['Infant mortality (per 1000 births)'].str.replace(',','.').astype('float')

data['Literacy (%)']=data['Literacy (%)'].str.replace(',','.').astype('float')

data['Phones (per 1000)']=data['Phones (per 1000)'].str.replace(',','.').astype('float')

data['Arable (%)']=data['Arable (%)'].str.replace(',','.').astype('float')

data['Crops (%)']=data['Crops (%)'].str.replace(',','.').astype('float')

data['Other (%)']=data['Other (%)'].str.replace(',','.').astype('float')

data['Birthrate']=data['Birthrate'].str.replace(',','.').astype('float')

data['Deathrate']=data['Deathrate'].str.replace(',','.').astype('float')

data['Agriculture']=data['Agriculture'].str.replace(',','.').astype('float')

data['Industry']=data['Industry'].str.replace(',','.').astype('float')

data['Service']=data['Service'].str.replace(',','.').astype('float')

data['Region']=data['Region'].str.strip()

data['Country']=data['Country'].str.strip()
data_copy=data.copy()
for x,y in data_copy[data_copy['Climate'].isnull()].iterrows():

    climate_value=data_copy.loc[data_copy['Region']==y['Region']]['Climate'].value_counts().index[0]

    data_copy.at[x,'Climate']=climate_value



for x,y in data_copy[data_copy['Net migration'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Net migration'].mean()

    data_copy.at[x,'Net migration']=value



for x,y in data_copy[data_copy['Infant mortality (per 1000 births)'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Infant mortality (per 1000 births)'].mean()

    data_copy.at[x,'Infant mortality (per 1000 births)']=value



for x,y in data_copy[data_copy['Literacy (%)'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Literacy (%)'].mean()

    data_copy.at[x,'Literacy (%)']=value



for x,y in data_copy[data_copy['Phones (per 1000)'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Phones (per 1000)'].mean()

    data_copy.at[x,'Phones (per 1000)']=value



for x,y in data_copy[data_copy['Arable (%)'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Arable (%)'].mean()

    data_copy.at[x,'Arable (%)']=value



for x,y in data_copy[data_copy['Crops (%)'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Crops (%)'].mean()

    data_copy.at[x,'Crops (%)']=value



for x,y in data_copy[data_copy['Other (%)'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Other (%)'].mean()

    data_copy.at[x,'Other (%)']=value

    

for x,y in data_copy[data_copy['Birthrate'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Birthrate'].mean()

    data_copy.at[x,'Birthrate']=value

    

for x,y in data_copy[data_copy['Deathrate'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Deathrate'].mean()

    data_copy.at[x,'Deathrate']=value

    

for x,y in data_copy[data_copy['Agriculture'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Agriculture'].mean()

    data_copy.at[x,'Agriculture']=value

                

for x,y in data_copy[data_copy['Industry'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Industry'].mean()

    data_copy.at[x,'Industry']=value

                

for x,y in data_copy[data_copy['Service'].isnull()].iterrows():

    value=data_copy.loc[data_copy['Region']==y['Region']]['Service'].mean()

    data_copy.at[x,'Service']=value
data_copy=data_copy.dropna(how='any')
data_copy['Region']=data_copy['Region'].astype('category')
data_copy['Climate']=data_copy['Climate'].astype('category')
data_copy.loc[80,'Pop. Density (per sq. mi.)']=data_copy.loc[80,'Population']/data_copy.loc[80,'Area (sq. mi.)']
fig, ax = plt.subplots(figsize=(12,12))

sns.heatmap(data_copy.corr(),annot=True)

plt.show()
sns.countplot(data_copy['Region'])

plt.xticks(rotation='vertical')

plt.show()
sns.countplot(data_copy['Climate'])

plt.xticks(rotation='vertical')

plt.show()
data_copy.columns
sns.boxplot(data_copy['Pop. Density (per sq. mi.)'])
sns.boxplot(data_copy['Coastline (coast/area ratio)'])
sns.distplot(data_copy['Coastline (coast/area ratio)'])
sns.distplot(data_copy['Net migration'])

plt.show()

print(data_copy['Net migration'].skew())
sns.distplot(data_copy['Infant mortality (per 1000 births)'])
sns.boxplot(data_copy['Infant mortality (per 1000 births)'])
sns.distplot(data_copy['GDP ($ per capita)'])
sns.boxplot(data_copy['GDP ($ per capita)'])
sns.distplot(data_copy['Literacy (%)'])
sns.distplot(data_copy['Phones (per 1000)'])
sns.distplot(data_copy['Crops (%)'])
sns.distplot(data_copy['Other (%)'])
sns.distplot(data_copy['Birthrate'])
sns.distplot(data_copy['Deathrate'])
sns.distplot(data_copy['Agriculture'])
sns.distplot(data_copy['Industry'])
sns.distplot(data_copy['Service'])
sns.lmplot(x='Literacy (%)',y='GDP ($ per capita)',data=data_copy)
sns.lmplot(x='Phones (per 1000)',y='GDP ($ per capita)',data=data_copy)
sns.lmplot(x='Service',y='GDP ($ per capita)',data=data_copy)
sns.lmplot(x='Birthrate',y='Infant mortality (per 1000 births)',data=data_copy)

sns.lmplot(x='Deathrate',y='Infant mortality (per 1000 births)',data=data_copy)