# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/master.csv')

df.sample(10)
df.info()
df.describe()
sn.heatmap(df.corr(),annot=True,cmap='RdYlBu')

plt.show()
#country year and HDI(null) are of no use

to_drop = df[['country-year','HDI for year']]

df = df.drop(to_drop,axis=1)

df.columns
data_country=df[(df['year']==2010)]



country_2010_population=[]

country_2010=df[(df['year']==2010)].country.unique()



for country in country_2010:

    country_2010_population.append(sum(data_country[(data_country['country']==country)].population))    



#Now year 2010 find sum population every country



plt.figure(figsize=(10,18))

sn.barplot(y=country_2010,x=country_2010_population)

plt.xlabel('Population Count')

plt.ylabel('Countries')

plt.title('2010 Sum Population for Suicide Rate')

plt.show()
data_country=df[(df['year']==2016)]



country_2016_population=[]

country_2016=df[(df['year']==2016)].country.unique()



for country in country_2016:

    country_2016_population.append(sum(data_country[(data_country['country']==country)].population))    



#Now year 2016 find sum population every country



plt.figure(figsize=(10,18))

sn.barplot(y=country_2016,x=country_2016_population)

plt.xlabel('Population Count')

plt.ylabel('Countries')

plt.title('2016 Year Sum Population for Suicide Rate')

plt.show()
pie_chart = plt.pie(df.generation.value_counts(),explode=[0.1,0.1,0.1,0.1,0.1,0.1],autopct='%0.1f%%')

plt.legend(df.generation)

plt.show(pie_chart)

suicides = df['suicides/100k pop']

sn.jointplot(x=df.year,y=suicides,data=df)

plt.show()
df.sex.value_counts()
suicides = df['suicides_no']

year = df['year']

plt.figure(figsize=(10,20))

sn.jointplot(x=year,y=suicides,data=df)

plt.show()
sn.countplot(df['sex'])

plt.show()
sn.countplot(df['generation'])

plt.xticks(rotation=45)

plt.show()
sn.pairplot(df,hue="sex")

plt.show()
sn.pairplot(df,hue="age")

plt.show()
print(df.age.value_counts())

sn.countplot(df.age)

plt.xticks(rotation=45)

plt.show()