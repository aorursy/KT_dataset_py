# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv',index_col='id')

data.sample(5)
data.info()
plt.figure(figsize=(14,8))

sns.countplot(data.sort_values('neighbourhood_group').neighbourhood_group,palette='Set2',alpha=0.8)

plt.title('Borough wise Airbnb listings in NYC')

plt.xlabel('Borough name')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(14,8))

sns.countplot(data.sort_values('room_type').room_type,palette='Set2')

plt.title('Room type count')

plt.xlabel('Room type')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(14,8))

sns.countplot(data.sort_values('neighbourhood_group').neighbourhood_group,hue=data.room_type,palette='Set2')

plt.title('Borough wise room type count')

plt.xlabel('Borough name')

plt.ylabel('Count')

plt.show()
plt.figure(figsize=(14,8))

sns.distplot(data.price)

plt.title('Distribution of price')

plt.show()
if np.mean(data.price)-np.std(data.price)<0:

    count=0

else:

    count=np.mean(data.price)-np.std(data.price)

print('Most of prices lies between the range: $' + str(count) + ' to $' + str(round(np.mean(data.price)+np.std(data.price))))
plt.figure(figsize=(14,8))

sns.distplot(data[data.price<1000].price)

plt.title('Distribution of price (only where price<1000)')

plt.show()
print(np.mean(data[data.price<1000].price)-np.std(data[data.price<1000].price),np.mean(data[data.price<1000].price)+np.std(data[data.price<1000].price))
plt.figure(figsize=(14,8))

sns.distplot(data[data.neighbourhood_group=='Manhattan'].price,color='maroon',hist=False,label='Manhattan')

sns.distplot(data[data.neighbourhood_group=='Brooklyn'].price,color='black',hist=False,label='Brooklyn')

sns.distplot(data[data.neighbourhood_group=='Queens'].price,color='green',hist=False,label='Queens')

sns.distplot(data[data.neighbourhood_group=='Staten Island'].price,color='blue',hist=False,label='Staten Island')

sns.distplot(data[data.neighbourhood_group=='Long Island'].price,color='lavender',hist=False,label='Long Island')

plt.title('Borough wise price destribution for price<2000')

plt.xlim(0,2000)

plt.show()

plt.figure(figsize=(14,8))

sns.distplot(data.minimum_nights).set_yscale('log')

plt.title('Minimum no. of nights distribution')

plt.show()

plt.figure(figsize=(14,8))

sns.distplot(data.reviews_per_month.dropna())

plt.title('Distribution of no. reviews per month')

plt.show()

plt.figure(figsize=(14,8))

sns.distplot(data.availability_365)

plt.title('Distribution of availability')

plt.show()
plt.figure(figsize=(14,8))

sns.boxplot(data.neighbourhood_group,data.price)

plt.ylim(0,2000)

plt.show()

import folium

from folium.plugins import HeatMap
m=folium.Map([40.7128,-74.0060],zoom_start=11)

HeatMap(data[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)

data.corr().style.background_gradient(cmap='coolwarm')

#No strong correlation except number_of_reviews vs reviews_per_month