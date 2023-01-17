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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
country=pd.read_csv('/kaggle/input/countries-of-the-world/countries of the world.csv')
country.head()
country.info()
country.describe()
a=country['Pop. Density (per sq. mi.)'].str.replace(',', '.')
b=a.astype(float)
country['Pop. Density (per sq. mi.)']=b
c=country['Coastline (coast/area ratio)'].str.replace(',', '.')
d=c.astype(float)
country['Coastline (coast/area ratio)']=d
c=country['Net migration'].str.replace(',', '.')
d=c.astype(float)
country['Net migration']=d
c=country['Infant mortality (per 1000 births)'].str.replace(',', '.')
d=c.astype(float)
country['Infant mortality (per 1000 births)']=d
c=country['Literacy (%)'].str.replace(',', '.')
d=c.astype(float)
country['Literacy (%)']=d
c=country['Phones (per 1000)'].str.replace(',', '.')
d=c.astype(float)
country['Phones (per 1000)']=d
c=country['Arable (%)'].str.replace(',', '.')
d=c.astype(float)
country['Arable (%)']=d
c=country['Crops (%)'].str.replace(',', '.')
d=c.astype(float)
country['Crops (%)']=d
c=country['Other (%)'].str.replace(',', '.')
d=c.astype(float)
country['Other (%)']=d
c=country['Climate'].str.replace(',', '.')
d=c.astype(float)
country['Climate']=d
e=country['Birthrate'].str.replace(',', '.')
f=e.astype(float)
country['Birthrate']=f
g=country['Deathrate'].str.replace(',', '.')
h=g.astype(float)
country['Deathrate']=h
g=country['Agriculture'].str.replace(',', '.')
h=g.astype(float)
country['Agriculture']=h
g=country['Industry'].str.replace(',', '.')
h=g.astype(float)
country['Industry']=h
g=country['Service'].str.replace(',', '.')
h=g.astype(float)
country['Service']=h
f=country['Climate'].quantile(0.75)
country['Climate'].fillna(f,inplace=True)
c=country['GDP ($ per capita)'].min()
d=country['GDP ($ per capita)'].fillna(c,inplace=True)
#replacing with the minimum value because the region is financially weak.
a=country['Net migration'].min()
b=country['Net migration'].fillna(a,inplace=True)
#replacing the nan with the lowest value of net migration because those regions are very difficult to live
yes=country['Industry'].quantile(0.75)
country['Industry'].fillna(yes,inplace=True)
hm=country['Literacy (%)'].mean()
country['Literacy (%)'].fillna(hm,inplace=True)
yas=country['Infant mortality (per 1000 births)'].mean()
country['Infant mortality (per 1000 births)'].fillna(yas,inplace=True)
y=country['Agriculture'].mean()
country['Agriculture'].fillna(y,inplace=True)
hi=country['Service'].quantile(0.5)
country['Service'].fillna(hi,inplace=True)
a=country['Birthrate'].mean()
b=country['Deathrate'].quantile(0.5)
country['Birthrate'].fillna(a,inplace=True)
country['Deathrate'].fillna(b,inplace=True)
a=country['Phones (per 1000)'].mean()
b=country['Arable (%)'].mean()
c=country['Crops (%)'].mean()
d=country['Other (%)'].mean()
country['Phones (per 1000)'].fillna(a,inplace=True)
country['Arable (%)'].fillna(b,inplace=True)
country['Crops (%)'].fillna(c,inplace=True)
country['Other (%)'].fillna(d,inplace=True)
country.info()
country.describe()
country['Climate']=country['Climate'].astype('category')
country=country.set_index('Country')
country['Population'].sort_values(ascending=False).head(5)

#conclusion: five most populated country.
country['Area (sq. mi.)'].sort_values(ascending=False).head(5)

#conclusion: These are the five most largest countries.
country['Pop. Density (per sq. mi.)'].sort_values(ascending=False).head(5)

#conclusion: Five most crowded countries.
country[country['Climate']==3].index #observing..
country[country['Climate']==1].index #observing..
country[country['Climate']==2].index #observing..
country[country['Climate']==1.5].index
#conclusion:There is an error because the climate of china is not the same as in those african countries in real life.
country[country['Climate']==2.5].index
#error because the geographical position of india ar swaziland is way different,so the climate of these two shouldnt be the same.
sns.countplot(country['Climate'])

#conclusion: most of the countries have hot weather as their climate or the avg temperature is high.
a=country['Region'].value_counts()
a
#regions in world
sns.boxplot(country['Birthrate'])
#Conclusion: Outliers are not there.
sns.distplot(country['Birthrate'])
country['Birthrate'].skew()
#Most of the countries have birthrates between 10 to 20.
sns.distplot(country['Net migration'])
country['Net migration'].skew()

sns.boxplot(country['Net migration'])
#Outliers exist in this column.
sns.distplot(country['Area (sq. mi.)'])
sns.boxplot(country['Area (sq. mi.)'])
sns.boxplot(country['Literacy (%)'])
sns.distplot(country['Literacy (%)'])
sns.boxplot(country['Industry'])
sns.boxplot(country['Service'])
sns.boxplot(country['Agriculture'])
sns.boxplot(country['GDP ($ per capita)'])
print("top 5 with most gdp",country['GDP ($ per capita)'].sort_values(ascending=False).head())
#Conclusion:Outliers arent there,most of the gdp lies within 15000. 
sns.distplot(country['GDP ($ per capita)'])
sns.boxplot(country['Phones (per 1000)'])
#Conclusion: Outliers dont exist.
sns.distplot(country['Phones (per 1000)'])
print(country['Phones (per 1000)'].skew())
#Conclusion: skewed data.
#Out of Curiosity:
print("Literacy rate below 30 is",country[country['Literacy (%)']<30].shape[0])
print("Net Migration lower than -20 is",country[country['Net migration']<-20].shape[0])
print("Net Migration greater than +20 is",country[country['Net migration']>20].shape[0])
print("GDP greater than 35000 is",country[country['GDP ($ per capita)']>35000].shape[0])
sns.distplot(country[country['Climate']==1]['Birthrate'])
sns.distplot(country[country['Climate']==2]['Birthrate'])
sns.distplot(country[country['Climate']==3]['Birthrate'])
sns.distplot(country[country['Climate']==1]['Literacy (%)'])
sns.distplot(country[country['Climate']==2]['Literacy (%)'])
sns.distplot(country[country['Climate']==3]['Literacy (%)'])
#if climate=3 or if the avg. temperature of a certain country is cold/low,then the chances of higher literacy rate is highest.
sns.distplot(country[country['Climate']==1]['Birthrate'])
sns.distplot(country[country['Climate']==2]['Birthrate'])
sns.distplot(country[country['Climate']==3]['Birthrate'])
sns.distplot(country[country['Climate']==1]['Deathrate'])
sns.distplot(country[country['Climate']==2]['Deathrate'])
sns.distplot(country[country['Climate']==3]['Deathrate'])
sns.heatmap(country.corr())
Q1=np.percentile(country['Literacy (%)'],25)
Q3=np.percentile(country['Literacy (%)'],75)
low=Q1-1.5*(Q3-Q1)
high=Q3+1.5*(Q3-Q1)
country=country[(country['Literacy (%)']>low) & (country['Literacy (%)']<high)]
Q1=np.percentile(country['Phones (per 1000)'],25)
Q3=np.percentile(country['Phones (per 1000)'],75)
low=Q1-1.5*(Q3-Q1)
high=Q3+1.5*(Q3-Q1)
country=country[(country['Phones (per 1000)']>low) & (country['Phones (per 1000)']<high)]
Q1=np.percentile(country['Agriculture'],25)
Q3=np.percentile(country['Agriculture'],75)
low=Q1-1.5*(Q3-Q1)
high=Q3+1.5*(Q3-Q1)
country=country[(country['Agriculture']>low) & (country['Agriculture']<high)]
country.info()
country['Net Birthrate']=country['Birthrate']-country['Deathrate']
country.drop(['Birthrate','Deathrate'],axis=1,inplace=True)
country.info()
country['Development']=country['Agriculture']+country['Industry']+country['Service']
country.drop(['Agriculture','Service','Industry'],axis=1,inplace=True)
country=country.reset_index()
country.info()