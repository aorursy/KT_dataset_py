# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/master.csv')
data.head()
data.dtypes
data.columns
data.columns = ['country', 'year', 'sex', 'age', 'suicides_no', 'population',

       'suicides/100k_pop', 'country-year', 'HDI_for_year',

       'gdp_for_year($)', 'gdp_per_capita($)', 'generation']
data.head()
data['year'] = data['year'].astype('object')
data['gdp_for_year($)'] = data['gdp_for_year($)'].str.replace(',','').astype('Int64')
data.head()
data.dtypes
data.info()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(20,50))

y = data['country']

sns.set_context("paper", 2.5, {"lines.linewidth": 4})

sns.countplot(y=y,label='count')
plt.figure(figsize=(30,10))

y = data['year']

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y,label='count')
y = data['sex']

sns.set_context("paper", 2.5, {"lines.linewidth":4})

sns.countplot(y,label='count')
plt.figure(figsize=(15,5))

y = data['age']

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y,label='count',order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(15,5))

y = data['generation']

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.countplot(y,label='generation',order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(30,10))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.barplot(data=data,x='year',y='suicides_no',hue='sex')
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']<1996],x='year',y='suicides_no',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >1995) & (data['year']<2006)],x='year',y='suicides_no',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >2005)],x='year',y='suicides_no',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']<1996],x='year',y='suicides_no',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >1995) & (data['year']<2006)],x='year',y='suicides_no',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']>2005],x='year',y='suicides_no',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
data[(data['year'] == 2010) & (data['generation'] == 'Boomers')]
data[(data['year'] == 2009) & (data['generation'] == 'Boomers')].head()
data[(data['year'] == 2011) & (data['generation'] == 'Boomers')].head()
print('year:2009,generation:G.I. Generation ',np.unique(data[(data['year'] == 2009) & (data['generation'] == 'G.I. Generation')]['age'].values))

print('year:2009,generation:Silent ',np.unique(data[(data['year'] == 2009) & (data['generation'] == 'Silent')]['age'].values))

print('year:2009,generation:Boomers ',np.unique(data[(data['year'] == 2009) & (data['generation'] == 'Boomers')]['age'].values))

print('year:2009,generation:Generation X ',np.unique(data[(data['year'] == 2009) & (data['generation'] == 'Generation X')]['age'].values))

print('year:2009,generation:Millenials ',np.unique(data[(data['year'] == 2009) & (data['generation'] == 'Millenials')]['age'].values))

print('year:2009,generation:Generation Z ',np.unique(data[(data['year'] == 2009) & (data['generation'] == 'Generation Z')]['age'].values))

print('-'*100)

print('year:2010,generation:G.I. Generation ',np.unique(data[(data['year'] == 2010) & (data['generation'] == 'G.I. Generation')]['age'].values))

print('year:2010,generation:Silent ',np.unique(data[(data['year'] == 2010) & (data['generation'] == 'Silent')]['age'].values))

print('year:2010,generation:Boomers ',np.unique(data[(data['year'] == 2010) & (data['generation'] == 'Boomers')]['age'].values))

print('year:2010,generation:Generation X ',np.unique(data[(data['year'] == 2010) & (data['generation'] == 'Generation X')]['age'].values))

print('year:2010,generation:Millenials ',np.unique(data[(data['year'] == 2010) & (data['generation'] == 'Millenials')]['age'].values))

print('year:2010,generation:Generation Z ',np.unique(data[(data['year'] == 2010) & (data['generation'] == 'Generation Z')]['age'].values))

print('-'*100)

print('year:2011,generation:G.I. Generation ',np.unique(data[(data['year'] == 2011) & (data['generation'] == 'G.I. Generation')]['age'].values))

print('year:2011,generation:Silent ',np.unique(data[(data['year'] == 2011) & (data['generation'] == 'Silent')]['age'].values))

print('year:2011,generation:Boomers ',np.unique(data[(data['year'] == 2011) & (data['generation'] == 'Boomers')]['age'].values))

print('year:2011,generation:Generation X ',np.unique(data[(data['year'] == 2011) & (data['generation'] == 'Generation X')]['age'].values))

print('year:2011,generation:Millenials ',np.unique(data[(data['year'] == 2011) & (data['generation'] == 'Millenials')]['age'].values))

print('year:2011,generation:Generation Z ',np.unique(data[(data['year'] == 2011) & (data['generation'] == 'Generation Z')]['age'].values))
year_list = list(range(1985,2017))

for i in year_list:

    print('year:{},generation:G.I. Generation {}'.format(i,np.unique(data[(data['year'] == i) & (data['generation'] == 'G.I. Generation')]['age'].values)))

    print('year:{},generation:Silent {}'.format(i,np.unique(data[(data['year'] == i) & (data['generation'] == 'Silent')]['age'].values)))

    print('year:{},generation:Boomers {}'.format(i,np.unique(data[(data['year'] == i) & (data['generation'] == 'Boomers')]['age'].values)))

    print('year:{},generation:Generation X {}'.format(i,np.unique(data[(data['year'] == i) & (data['generation'] == 'Generation X')]['age'].values)))

    print('year:{},generation:Millenials {}'.format(i,np.unique(data[(data['year'] == i) & (data['generation'] == 'Millenials')]['age'].values)))

    print('year:{},generation:Generation Z {}'.format(i,np.unique(data[(data['year'] == i) & (data['generation'] == 'Generation Z')]['age'].values)))

    print('-'*100)
plt.figure(figsize=(30,10))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.barplot(data=data,x='year',y='population',hue='sex')
plt.figure(figsize=(30,10))

sns.barplot(data=data[data['year']<1996],x='year',y='population',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >1995) & (data['year']<2006)],x='year',y='population',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >2005)],x='year',y='population',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']<1996],x='year',y='population',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >1995) & (data['year']<2006)],x='year',y='population',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']>2005],x='year',y='population',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(30,10))

sns.set_context("paper", 2.0, {"lines.linewidth": 4})

sns.barplot(data=data,x='year',y='suicides/100k_pop',hue='sex')
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']<1996],x='year',y='suicides/100k_pop',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >1995) & (data['year']<2006)],x='year',y='suicides/100k_pop',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >2005)],x='year',y='suicides/100k_pop',hue='age',hue_order=['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']<1996],x='year',y='suicides/100k_pop',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[(data['year'] >1995) & (data['year']<2006)],x='year',y='suicides/100k_pop',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(20,10))

sns.barplot(data=data[data['year']>2005],x='year',y='suicides/100k_pop',hue='generation',hue_order=['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])
plt.figure(figsize=(20,50))

sns.set_context("paper", 2.5, {"lines.linewidth": 4})

sns.barplot(data=data[data['year']>2005],x='HDI_for_year',y='country',hue='year')
plt.figure(figsize=(20,50))

sns.set_context("paper", 2.5, {"lines.linewidth": 4})

sns.barplot(data=data[data['year']>2005],x='gdp_for_year($)',y='country',hue='year')
plt.figure(figsize=(20,50))

sns.set_context("paper", 2.5, {"lines.linewidth": 4})

sns.barplot(data=data[data['year']>2005],x='gdp_per_capita($)',y='country',hue='year')
data_scaled = data.loc[:,['HDI_for_year','suicides/100k_pop']]

data_scaled = (data_scaled - data_scaled.mean()) / data_scaled.std()

plt.figure(figsize=(10,10))

sns.scatterplot(data=data_scaled,x='HDI_for_year',y='suicides/100k_pop')
data_scaled = data.loc[:,['gdp_for_year($)','suicides/100k_pop']]

data_scaled = (data_scaled - data_scaled.mean()) / data_scaled.std()

sns.scatterplot(data=data_scaled,x='gdp_for_year($)',y='suicides/100k_pop')
data_scaled = data.loc[:,['gdp_per_capita($)','suicides/100k_pop']]

data_scaled = (data_scaled - data_scaled.mean()) / data_scaled.std()

sns.scatterplot(data=data_scaled,x='gdp_per_capita($)',y='suicides/100k_pop')