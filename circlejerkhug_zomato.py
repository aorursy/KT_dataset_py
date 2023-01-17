# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode

init_notebook_mode(connected=False)

from wordcloud import WordCloud

from folium.plugins import HeatMap

import folium

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

data=pd.read_csv('../input/zomato.csv')

data.info()
data.head()
data.drop('url',axis=1,inplace=True)

data.drop('phone',axis=1,inplace=True)
data.tail()
data['rate'] = data['rate'].replace('NEW',np.NaN)

data['rate'] = data['rate'].replace('-',np.NaN)

data.dropna(how = 'any', inplace = True)
data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)

data['rate'] = data['rate'].astype(str)

data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))

data['rate'] = data['rate'].apply(lambda r: float(r))
data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',','')

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(int)
data.head()
loc=data.loc[:, "location"].unique()

print(loc)

loc.sort()
def show_articles_more_than(Restaurant_Name=''):

    return data[data['name'].str.contains(Restaurant_Name)]
def show_Restaurants_according_to_search(Location=loc,

                                         Restaurant_Type=['Buffet', 

                                             'Cafes',

                                             'Delivery',

                                             'Desserts',

                                             'Dine-out',

                                             'Drinks & nightlife',

                                             'Pubs and bars'],

                            Min_Rating=(0,5,0.1),

                            Max_Cost_For_Two_People=(100,5000,50)):

    print("")

    return data[ (data['rate'] > Min_Rating) 

                &(data['listed_in(type)'] == Restaurant_Type) 

                &(data['location'] == Location) 

                & (data['approx_cost(for two people)'] < Max_Cost_For_Two_People)]
plt.figure(figsize=(10,7))

chains=data['name'].value_counts()[:10]

sns.barplot(x=chains,y=chains.index,palette='deep')

plt.title("Most famous restaurants chains in Bangaluru")

plt.xlabel("Number of outlets")
sns.countplot(x=data['online_order'])

fig = plt.gcf()

fig.set_size_inches(6,6)

plt.title('Restaurants delivering order online')
sns.countplot(x=data['online_order'],hue=data['listed_in(type)'])

fig=plt.gcf()

fig.set_size_inches(14,8)

sns.countplot(x=data['book_table'])

fig = plt.gcf()

fig.set_size_inches(6,6)

plt.title('Restaurants providing Table booking facility:')
sns.countplot(x=data['book_table'],hue=data['listed_in(type)'])

fig=plt.gcf()

fig.set_size_inches(14,8)

print("All different dining type restaurants")

data['listed_in(type)'].unique()
plt.figure(figsize=(20,7))

sns.countplot(x=data['rest_type'],hue=data['listed_in(type)'])

plt.title("Restaurant types")

plt.xlabel("count")
data.rate.unique()
sns.countplot(x=data['rate'])

fig = plt.gcf()

fig.set_size_inches(20,6)

plt.title('Rating of the resturants')
slices=[((data.rate>=0) & (data.rate<1)).sum(),

        ((data.rate>=1) & (data.rate<2)).sum(),

        ((data.rate>=2) & (data.rate<3)).sum(),

        ((data.rate>=3.0) & (data.rate<4)).sum(),

        ((data.rate>=4) & (data.rate<5)).sum(),

       ]

labels=['0-1','1-2','2-3','3-4','4-5']

plt.pie(slices, labels=labels, autopct='%1.0f%%', pctdistance=.5, labeldistance=1.2,shadow=True)

fig = plt.gcf()

plt.title("Percentage of restaurants according to their ratings")



fig.set_size_inches(10,10)

plt.show()
plt.figure(figsize=(20,10))

ax = sns.countplot(x='rate',hue='online_order',data=data)

plt.title('Rating of Restaurants vs Online Delivery')

plt.show()
plt.figure(figsize=(20,10))

ax = sns.countplot(x='rate',hue='book_table',data=data)

plt.title('Rating of Restaurants vs Table Booking')

plt.show()
data.plot(kind='scatter',x='rate',y='votes',marker='o',color='g',grid=True,figsize=(20,7))

plt.title('Votes Versus Rate');
data.plot(kind='scatter',x='rate',y='approx_cost(for two people)',marker='o',color='b',grid=True,figsize=(20,7))

plt.title('Average cost for Two Persons Versus Rate',weight='bold');
print("all different cuisines:")

cuisines = set()

for i in data['cuisines']:

    for j in str(i).split(', '):

        cuisines.add(j)

cuisines
import re

data=data[data['dish_liked'].notnull()]

data.index=range(data.shape[0])

likes=[]

for i in range(data.shape[0]):

    splited_array=re.split(',',data['dish_liked'][i])

    for item in splited_array:

        likes.append(item)

print("Count of Most liked dishes of Bangalore")

favourite_food = pd.Series(likes).value_counts()

favourite_food.head(20)
plt.figure(figsize=(7,7))

cuisines=data['cuisines'].value_counts()[:10]

sns.barplot(cuisines,cuisines.index,palette='rocket')

plt.xlabel('Count')

plt.title("Most popular cuisines of Bangalore")
locCount=data['location'].value_counts().sort_values(ascending=True)

locCount
CityCount=data['listed_in(city)'].value_counts().sort_values(ascending=True)

CityCount
plt.figure(figsize=(20,7))

data.groupby('location')['votes'].sum().sort_values(ascending=False)[:10].plot('bar',color='r')

plt.title('Top 10 Voted neighbourhoods',weight='bold')

plt.ylabel('Count');
