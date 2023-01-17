# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # package for plotting and viewing data

import seaborn as sns # advanced package for plotting and visualization



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
restaurants = pd.read_csv("../input/zomato.csv")

restaurants.head()
## Getting insights from dataset



restaurants.keys()
restaurants.shape
## Plotting categorical data to understand on type and facicilties 



## Restaurants accepting Online Order

sns.set_style("darkgrid")

sns.countplot(x="online_order",data=restaurants)
## Restaurants allowing to book table

sns.countplot(x="book_table",data=restaurants)
sns.countplot(x="listed_in(type)",data=restaurants)
## Restaurants that allow both table booking and online order

all_rest = restaurants.query('online_order == "Yes" and book_table == "Yes"')

print("There are {} restaurants in Bangalore that allows both table booking and online order from a list of {} restaurants".format(len(all_rest.index), len(restaurants.index)))
## Location where the above restaurants are located

restaurant_loc = all_rest.groupby('location',sort=False, as_index=False).count()

restaurant_loc = restaurant_loc.loc[:,("location","url")]

restaurant_loc.rename(columns={"url":"count"}, inplace=True)

restaurant_loc

plt.plot(restaurant_loc['location'],restaurant_loc['count'])

plt.show()
# Using seaborn to plot the bar chart of above data

plt.subplots(figsize=(6,15))

sns.barplot(y=restaurant_loc['location'],x=restaurant_loc['count'], data = restaurant_loc)
#Details for maximum number of restaurants allowing both online and table booking

restaurant_loc.iloc[restaurant_loc['count'].idxmax()]
## Type of restaurant

plt.subplots(figsize=(6,15))

sns.countplot(x="rest_type",data=restaurants)



## The above would be a plot of the whole dataset. Displaying this detail in a table.

restaurants.groupby("rest_type").size().sort_values(ascending=False)
# Number of Restaurants found across Bangalore



max_restaurant = restaurants.groupby(['listed_in(city)']).size().sort_values(ascending=False)

#plotting the above data

plt.subplots(figsize=(6,15))

sns.barplot(max_restaurant.values, max_restaurant.index)
# Most popular restaurant by number of visitors/reviews/votes



## working in cuisines

restaurants.groupby(['listed_in(city)','cuisines']).size().sort_values(ascending=False)
# Categorize restaurants based on cost.

restaurants.groupby(['listed_in(city)','approx_cost(for two people)','cuisines']).size().sort_values(ascending=False)
restaurants.loc[restaurants['approx_cost(for two people)'] == '3,000']

# Converting the price to numeric data type. In Pandas world, converting object to int/float type

## First replace comma in the numeric field and then convert that to numeric

restaurants['approx_cost(for two people)'] = restaurants['approx_cost(for two people)'].str.replace(',','')

restaurants['approx_cost(for two people)'] = restaurants['approx_cost(for two people)'].astype(float)

print(restaurants.dtypes)
# Details of the most costly restaurant

costly_restaurant=restaurants.loc[restaurants['approx_cost(for two people)'].idxmax()]

print("The Most COSTLY restaurant in Bangalore is located at {}, {} and the approx cost for 2 people is Rs {}.It is none other than {}"

      .format(costly_restaurant['address'],costly_restaurant['listed_in(city)'], costly_restaurant['approx_cost(for two people)'], costly_restaurant['name']))
# Creating a geolocation of restaurants

import geopy

dir(geopy)
# Creating a word cloud for the dishes liked

import wordcloud
# Checking to see if there are nulls

restaurants['dish_liked'].isnull().sum()
# Dropping the above 28K nulls

restaurants['dish_liked'].dropna(inplace=True)
#Getting the list of dishes like

dishes_liked=[]

for dishes in restaurants['dish_liked']:

    for dish in dishes.split(', '):

        dishes_liked.append(dish)

    

len(dishes_liked)



# There are 128K dishes
unique_dishes=list(set(dishes_liked))
len(np.unique(unique_dishes))
#Creating a disctionary of dishes liked and their counts using numpy package

unique, counts = np.unique(dishes_liked, return_counts=True)

dish_frequency = dict(zip(unique, counts))



## Getting an ordered dictionary

from collections import OrderedDict as od

ordered_dish_frequency = od(sorted(dish_frequency.items(), key = lambda x:x[1], reverse =True))

ordered_dish_frequency
# Building word cloud

wc= wordcloud.WordCloud().generate_from_frequencies(ordered_dish_frequency)



#Plotting the wordcloud

plt.figure(figsize=(20,20))

plt.imshow(wc, interpolation='bilinear')

plt.axis=("off")

plt.show()