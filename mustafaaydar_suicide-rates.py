# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

import seaborn as sns

from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

data.info()

data.describe()
data.fillna(0, inplace=True)

data.head()
data.corr()
f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
data.rename(columns={'suicides/100k pop': 'suicides_100k_pop', 'gdp_per_capita ($)': 'gdp_per_capita',' gdp_for_year ($) ': 'gdp_for_year','HDI for year':'HDI_for_year'}, inplace=True)

data.columns
# Line Plot

data.population.plot(kind = 'line', color = 'g',label = 'year',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.suicides_no.plot(color = 'r',label = 'suicides_no',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

data.plot(kind='scatter', x='suicides_100k_pop', y='gdp_per_capita',alpha = 0.5,color = 'red')

plt.xlabel('suicides_100k_pop')              # label = name of label

plt.ylabel('gdp_per_capita')

plt.title('Suicide/per_capita Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.suicides_no.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
data.suicides_no.plot(kind = 'hist',bins = 50)

plt.clf()
data['country'].unique()

state_list = list(data['country'].unique())

state_suicides_ratio = []

for i in state_list:

    x = data[data['country']==i]

    state_suicides_rate = sum(x.suicides_no)/len(x)

    state_suicides_ratio.append(state_suicides_rate)

new_data = pd.DataFrame({'state_list': state_list,'state_suicides_ratio':state_suicides_ratio})

new_index = (new_data['state_suicides_ratio'].sort_values(ascending=False)).index.values

sorted_data = new_data.reindex(new_index)

plt.figure(figsize=(45,15))

sns.barplot(x=sorted_data['state_list'], y=sorted_data['state_suicides_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Suicide Rate')

plt.title('Suicide Rate Given States')

data['year'].unique()

year_list = list(data['year'].unique())

year_suicides_amount = []

for i in state_list:

    x = data[data['year']==i]

    state_suicides_total = sum(x.suicides_no)

    year_suicides_amount.append(state_suicides_total)

new_data = pd.DataFrame({'year_list': year_list,'year_suicides_amount':year_suicides_amount})

new_index = (new_data['year_suicides_amount'].sort_values(ascending=False)).index.values

sorted_data = new_data.reindex(new_index)

print(year_suicides_amount)

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['year_list'], y=sorted_data['year_suicides_amount'])

plt.xticks(rotation= 45)

plt.xlabel('Years')

plt.ylabel('Suicide Mounts')

plt.title('Suicide Amounts Given Years')
data['age'].unique()

age_list = list(data['age'].unique())

state_suicides_ratio = []

for i in age_list:

    x = data[data['age']==i]

    state_suicides_rate = sum(x.suicides_no)

    state_suicides_ratio.append(state_suicides_rate)

new_data = pd.DataFrame({'age_list': age_list,'state_suicides_ratio':state_suicides_ratio})

new_index = (new_data['state_suicides_ratio'].sort_values(ascending=False)).index.values

sorted_data = new_data.reindex(new_index)



plt.figure(figsize=(25,15))

sns.barplot(x=sorted_data['age_list'], y=sorted_data['state_suicides_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Ages')

plt.ylabel('Suicide Amounts')

plt.title('Suicide Amounts Given Ages')

data['generation'].unique()

generation_list = list(data['generation'].unique())

state_suicides_ratio = []

for i in generation_list:

    x = data[data['generation']==i]

    state_suicides_rate = sum(x.suicides_no)/len(x)

    state_suicides_ratio.append(state_suicides_rate)

new_data = pd.DataFrame({'generation_list': generation_list,'state_suicides_ratio':state_suicides_ratio})

new_index = (new_data['state_suicides_ratio'].sort_values(ascending=False)).index.values

sorted_data = new_data.reindex(new_index)



plt.figure(figsize=(25,15))

sns.barplot(x=sorted_data['generation_list'], y=sorted_data['state_suicides_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('Generations')

plt.ylabel('Suicide Rate')

plt.title('Suicide Rate Given Generations')

data.head()
# high school graduation rate vs Poverty rate of each state

sorted_data['state_suicides_ratio'] = sorted_data['state_suicides_ratio']/max( sorted_data['state_suicides_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])

data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

data.sort_values('area_poverty_ratio',inplace=True)

# visualize

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='state_list',y='state_suicides_ratio',data=data,color='lime',alpha=0.8)

sns.pointplot(x='state_list',y='area_highschool_ratio',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()


