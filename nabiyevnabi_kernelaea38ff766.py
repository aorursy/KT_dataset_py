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
raw_data=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

raw_data



raw_data.info()
null_values=raw_data[raw_data['HDI for year'].isna()==False]

null=null_values.year

import collections

collections.Counter(null)

countries=raw_data['country-year'].unique()

pd.options.display.max_columns=None

pd.options.display.max_rows=50

countries=pd.DataFrame(countries)

display(countries)
raw_data['country'].unique()
region=pd.Series

europe=[]

asia=[]

samerica=[]

namerica=[]

for country in raw_data['country']:

    if country==europe.any():

        region.append('Europe')

    elif country==asia.any():

        region.append('Asia')

    elif country==samerica.any():

        region.append('South America')

    else:

        region.append('North America')

raw_data=pd.concat([raw_data,region],axis=1)    
import matplotlib.pyplot as plt

plt.plot(raw_data['year'],raw_data['suicides/100k pop'])
plt.scatter(raw_data['year'],raw_data['suicides/100k pop'])
import seaborn as sns

sns.boxplot(x='year', y='suicides/100k pop', data=raw_data)
youngsters_data=raw_data[raw_data['age']==('5-14 years'or'15-24 years')]

sns.boxplot(x='year', y='suicides/100k pop', data=youngsters_data)
developed_countries=raw_data[raw_data['HDI for year']>0.700]

sns.boxplot(x='year', y='suicides/100k pop', data=developed_countries)
plt.scatter(developed_countries['year'],developed_countries['suicides/100k pop'])
underdeveloping_countries=raw_data[raw_data['HDI for year']<0.700]

sns.boxplot(x='year', y='suicides/100k pop', data=underdeveloping_countries)
developed_countries_young=youngsters_data[youngsters_data['HDI for year']>0.700]

sns.boxplot(x='year', y='suicides/100k pop', data=developed_countries_young)
needed_data=raw_data.drop(['country-year','country'],axis=1)

needed_data=needed_data[needed_data['year']!=(1993|1994|1995|1996|1997)]

needed_data                                              
needed_data['year']=='1995'

pd.options.display.max_rows=None

display(needed_data['year']=='1995')
pd.options.display.max_rows=100
plt.hist(needed_data['suicides/100k pop'])
needed_data.drop(['year'],axis=1,inplace=True)

needed_data=needed_data[needed_data['suicides/100k pop']<50]
satisfactory_data=needed_data.copy()
satisfactory_data
satisfactory_data['HDI for year'].isna().sum()
satisfactory_data['sex'].value_counts()
satisfactory_data['age'].value_counts()