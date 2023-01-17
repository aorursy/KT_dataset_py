# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1',

                          usecols=[0, 1, 2, 3, 8, 11,  29, 35,58, 82, 98, 101])

data = data.rename(

    columns={'eventid':'id', 'iyear':'year', 'imonth':'month', 'iday':'day',

             'country_txt':'country', 'provstate':'city', 'attacktype1_txt':'attack', 'targtype1_txt':'target',

             'gname':'group','weaptype1_txt':'weapon', 'nkill':'fatalities', 'nwound':'injuries'})



data = data[(data.country=='Turkey')]



data.head()
print('City with Highest Terrorist Attacks:',data['city'].value_counts().index[0], ' on ',data['year'].value_counts().index[0])

print('Top attack type:',data['weapon'].value_counts().index[0])

print('Maximum people killed in an attack are:',data['fatalities'].max(),'that took place in',data.loc[data['fatalities'].idxmax()].city, ' on ',data.loc[data['fatalities'].idxmax()].year)
plt.subplots(figsize=(18,6))

sns.countplot('year',data=data,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',10))

plt.xticks(rotation=60)

plt.title('Number Of Terrorist Activities Each Year')

plt.grid()

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('month',data=data,palette='inferno',edgecolor=sns.color_palette('dark',7),order=data['month'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities in each Month')

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot(data['target'],order=data['target'].value_counts()[:15].index,palette='inferno')

plt.xticks(rotation=60)

plt.title('Favorite Targets')

plt.show()
plt.subplots(figsize=(18,6))

sns.barplot(data['city'].value_counts()[:15].index,data['city'].value_counts()[:15].values,palette='inferno')

plt.title('Top Affected Countries')

plt.show()
plt.subplots(figsize=(18,6))

sns.barplot(data['city'].value_counts()[:-16:-1].index,data['city'].value_counts()[:-16:-1].values,palette='RdYlGn_r')

plt.title('Least Affected Countries')

plt.show()

plt.subplots(figsize=(20,6))

sns.countplot('attack',data=data,palette='inferno',order=data['attack'].value_counts().index)

plt.xticks(rotation=20)

plt.title('Attacking Methods by Terrorists')

plt.show()
data_city = data[(data.attack=='Bombing/Explosion')]

plt.subplots(figsize=(18,6))

sns.barplot(data_city['city'].value_counts()[:15].index,data_city['city'].value_counts()[:15].values,palette='inferno')

plt.title('Top Affected City with Bombing/Explosion Attacks')

plt.xticks(rotation=90)

plt.show()



sns.barplot(data['group'].value_counts()[:15].values,data['group'].value_counts()[:15].index,palette=('inferno'))

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.title('Terrorist Groups with Highest Terror Attacks')

plt.show()