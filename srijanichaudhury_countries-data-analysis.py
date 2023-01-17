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
df=pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')
data=df.copy()
data.rename(columns={'Area (sq. mi.)':'Area','Pop. Density (per sq. mi.)':'Population_Density','Coastline (coast/area ratio)':'Coastline','Net migration':'Net_migration','Infant mortality (per 1000 births)':'Infant_mortality','Literacy (%)':'Literacy','Phones (per 1000)':'Phones','Arable (%)':'Arabia','Crops (%)':'Crops','Other (%)':'Other','GDP ($ per capita)':'GDP'},inplace=True)

data['Population_Density']=data['Population_Density'].str.replace(",",".").astype(float)
data['Coastline']=data['Coastline'].str.replace(",",".").astype(float)
data['Net_migration']=data['Net_migration'].str.replace(",",".").astype(float)
data['Infant_mortality']=data['Infant_mortality'].str.replace(",",".").astype(float)
data['Literacy']=data['Literacy'].str.replace(",",".").astype(float)
data['Phones']=data['Phones'].str.replace(",",".").astype(float)
data['Arabia']=data['Arabia'].str.replace(",",".").astype(float)
data['Crops']=data['Crops'].str.replace(",",".").astype(float)
data['Other']=data['Other'].str.replace(",",".").astype(float)
data['Birthrate']=data['Birthrate'].str.replace(",",".").astype(float)
data['Deathrate']=data['Deathrate'].str.replace(",",".").astype(float)
data['Agriculture']=data['Agriculture'].str.replace(",",".").astype(float)
data['Industry']=data['Industry'].str.replace(",",".").astype(float)
data['Service']=data['Service'].str.replace(",",".").astype(float)

data['Literacy']=data['Literacy'].fillna(data['Literacy'].mean())
data['Climate']=data['Climate'].fillna(2)
data['Net_migration']=data['Net_migration'].fillna(data['Net_migration'].mean())

data['Infant_mortality']=data['Infant_mortality'].fillna(data['Infant_mortality'].mean())
data['GDP']=data['GDP'].fillna(data['GDP'].mean())

data['Phones']=data['Phones'].fillna(data['Phones'].mean())
data['Arabia']=data['Arabia'].fillna(data['Arabia'].mean())
data['Crops']=data['Crops'].fillna(data['Crops'].mean())
data['Other']=data['Other'].fillna(data['Other'].mean())
data['Birthrate']=data['Birthrate'].fillna(data['Birthrate'].mean())
data['Deathrate']=data['Deathrate'].fillna(data['Deathrate'].mean())
data['Agriculture']=data['Agriculture'].fillna(data['Agriculture'].mean())
data['Service']=data['Service'].fillna(data['Service'].mean())
data['Industry']=data['Industry'].fillna(data['Industry'].mean())
data
data['Region']=data['Region'].str.strip()
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('fivethirtyeight')
sns.countplot(y=data['Region'],data=data)

sns.countplot(x=data['Climate'],data=data)
f,size=plt.subplots(figsize=(15,10))
sns.heatmap(data.corr(),annot=True,cbar="%0.1f%%",ax=size)
data.info()
print(len(data))
print(len(data.drop_duplicates(subset='Country')))
gdp=data.copy()
print("The average GDP is {} .Maximum GDP {} and minimum GDP {}".format(gdp['GDP'].mean(),max(gdp['GDP']),min(gdp['GDP'])))

gdp=gdp[gdp['GDP']<gdp['GDP'].quantile(0.99)]

plt.figure(figsize=(15,8))
sns.distplot(gdp['GDP'])
print(gdp['GDP'].skew())
plt.figure(figsize=(15,8))
sns.boxplot(y='GDP',x='Climate',data=gdp)
plt.figure(figsize=(15,8))
sns.boxplot(x='GDP',y='Region',data=gdp)

sns.relplot(x='Service',y='GDP',kind='line',data=gdp)
sns.relplot(x='Literacy',y='GDP',kind='line',data=gdp)
sns.relplot(x='Infant_mortality',y='GDP',kind='line',data=gdp)
sns.relplot(x='Phones',y='GDP',kind='line',data=gdp)
           

gdp=gdp.sort_values('GDP',ascending=False)
gdp[['Country','GDP']].head()
np.percentile(gdp['GDP'],50)
infant=data.copy()
print("The average infant mortality is {}.Maximum infant mortality {} and minimum mortality {}".format(infant['Infant_mortality'].mean(),max(infant['Infant_mortality']),min(infant['Infant_mortality'])))
infant=data.copy()
infant=infant[infant['Infant_mortality']<infant['Infant_mortality'].quantile(0.99)]
plt.figure(figsize=(15,8))
sns.distplot(infant['Infant_mortality'])
print(infant['Infant_mortality'].skew())
plt.figure(figsize=(15,8))
sns.boxplot(y='Infant_mortality',x='Climate',data=infant)
plt.figure(figsize=(15,8))
sns.boxplot(x='Infant_mortality',y='Region',data=infant)
sns.relplot(x='Infant_mortality',y='Agriculture',kind='line',data=infant)
print(infant['Infant_mortality'].quantile(0.5))
infant.sort_values('Infant_mortality',ascending=False).head()[['Country','Infant_mortality']]
crops=data.copy()
print("The average crop production is {}% .Maximum crop production {}% and minimum crop production {}%".format(crops['Crops'].mean(),max(crops['Crops']),min(crops['Crops'])))
crops=data.copy()
crops=crops[crops['Crops']<crops['Crops'].quantile(0.99)]
plt.figure(figsize=(15,8))
sns.distplot(crops['Crops'])
print(crops['Crops'].skew())
plt.figure(figsize=(15,8))
sns.boxplot(y='Crops',x='Climate',data=crops)
plt.figure(figsize=(15,8))
sns.boxplot(x='Crops',y='Region',data=crops)
sns.relplot(y='Crops',x='Coastline',kind='line',data=crops)
sns.relplot(y='Crops',x='Deathrate',kind='line',data=crops)
sns.relplot(y='Crops',x='Industry',kind='line',data=crops)
np.percentile(crops['Crops'],50)
crops.sort_values('Crops',ascending=False).head()[['Country','Crops']]
industry=data.copy()
print("The average industries sector is {} .Maximum no of industries {} and minimum  {}".format(industry['Industry'].mean(),max(industry['Industry']),min(industry['Industry'])))
industry=industry[industry['Industry']<industry['Industry'].quantile(0.99)]
plt.figure(figsize=(15,8))
sns.distplot(industry['Industry'])
print(industry['Industry'].skew())
plt.figure(figsize=(15,8))
sns.boxplot(y='Industry',x='Climate',data=industry)
plt.figure(figsize=(15,8))
sns.boxplot(x='Industry',y='Region',data=industry)
sns.relplot(y='Industry',x='Agriculture',kind='line',data=industry)

np.percentile(industry['Industry'],50)
industry.sort_values('Industry',ascending=False).head()[['Country','Industry']]
literacy=data.copy()
print("The average literary percentage is {} .Maximum literacy {} and minimum  {}".format(literacy['Literacy'].mean(),max(literacy['Literacy']),min(literacy['Literacy'])))
literacy=literacy[literacy['Literacy']<literacy['Literacy'].quantile(0.99)]
plt.figure(figsize=(15,8))
sns.distplot(literacy['Literacy'])
print(literacy['Literacy'].skew())
plt.figure(figsize=(15,8))
sns.boxplot(x='Literacy',y='Region',data=literacy)
sns.relplot(x='Literacy',y='Phones',kind='line',data=literacy)
sns.relplot(x='Literacy',y='GDP',kind='line',data=literacy)
np.percentile(literacy['Literacy'],50)
literacy.sort_values('Literacy',ascending=False).head()[['Country','Literacy']]
