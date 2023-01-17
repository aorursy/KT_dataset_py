# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
#%% Data info

data.info()
data.describe()
#%% Data first 5 row

data.head()
# sum null

data.isnull().sum()
#Corelation

corr = data.corr()
#%% Corelation plot

plt.subplots(figsize=(15, 15))

sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.5f')

plt.xticks(rotation=90)

plt.yticks(rotation=0)

plt.show()
state = data.neighbourhood_group.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state.index,y=state.values) #optional example= (x = state[:20].index,y=state[:20].values) #data delimitation 20 row  

plt.title('Most home area',color = 'blue',fontsize=15)
#%% # Neighbourhood_group Price



area_list = list(data['neighbourhood_group'].unique())

area_price_ratio = []

area_frequented = []

for i in area_list:

    x = data[data['neighbourhood_group']==i]

    area_price = sum(x.price)/len(x)

    area_frequented.append(area_price)

    area_price_ratio.append(area_price)

data1 = pd.DataFrame({'area_list': area_list,'area_price_ratio':area_price_ratio})

new_index = (data1['area_price_ratio'].sort_values(ascending=False)).index.values

sorted_data = data1.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'][:30], y=sorted_data['area_price_ratio'][:30])

plt.xticks(rotation= 90)

plt.xlabel('Area')

plt.ylabel('Price')

plt.title('House prices by area')
#To Be Continued