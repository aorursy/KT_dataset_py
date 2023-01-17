# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#read Country Code csv file into dataframe

rest_country = pd.read_excel("../input/Country-Code.xlsx")



#rename all columns in dataframe

rest_country.columns=['country code','country']



print (rest_country.head())



#read zomato restaurants csv file into dataframe

rest_data=pd.read_csv('../input/zomato.csv',encoding='latin-1')

#renaming all columns to lowercase for easy access

rest_data.columns=[x.lower() for x in rest_data.columns]



print (rest_data.head())
#printing column names of rest_data dataframe

print(rest_data.columns)



#print tuple showing total rows and columns in rest_data dataframe

rest_data.shape
#Join 2 dataframes to have Country Name column in resulting DataFrame

rest_all_data = pd.merge(rest_data,rest_country,on='country code',how='inner')



#print tuple to ensure addition of new column Country Name

rest_all_data.shape
#Find which all countries data in the dataset

rest_all_data['country'].unique()
#Find out number of restaurant registered on Zomato across all countries

print(rest_all_data['country'].value_counts())



#plot bar graph

rest_all_data['country'].value_counts().plot(kind='bar',title='Total Restaurants On Zomato In Countries'

                                             ,figsize=(20,10),fontsize=20)





rest_india = rest_all_data[rest_all_data['country']=='India']



rest_india['country'].unique()

rest_india.head(10)
rest_india.shape
#Find out percentage of data comprises to each city in India dataset

#as_index option set False to consider city as column of dataframe rather than index

grouped_cities=rest_india.groupby('city',as_index=False)[['restaurant id']].count().sort_values(ascending=False,by='restaurant id')

grouped_cities['total'] = grouped_cities['restaurant id'].sum()

grouped_cities['percent'] = (grouped_cities['restaurant id']/grouped_cities['total'])*100

#plot the Pie Chart showing percentage of restaurants in top 3 cities

colors = ['b', 'g', 'r']

explode = (0, 0, 0.3)

label = grouped_cities['city'].head(3)

values = grouped_cities['percent'].head(3).round(2)

plt.pie(values, colors=colors, labels= values ,explode=explode,counterclock=False, shadow=True)

plt.title('Percentage of Resturants on Zomato in top 3 cities of India')

plt.legend(label,loc=4)

plt.show()
rest_new_delhi = rest_india[rest_india['city'] == 'New Delhi']

rest_new_delhi.head()
coffee_shops = ['Costa Coffee','Starbucks','Barista','Cafe Coffee Day']



delhi_coffee_shops = rest_new_delhi[rest_new_delhi['restaurant name'].isin(coffee_shops)]

delhi_coffee_shops = delhi_coffee_shops.groupby('restaurant name',as_index=False)[['aggregate rating','average cost for two']].mean().round(2).sort_values(ascending=False,by='aggregate rating')

#bar graph to plot average cost for 2 people

costs = delhi_coffee_shops['average cost for two']

rnames = delhi_coffee_shops['restaurant name']

colors = ['g','b','r','y']

plt.bar(rnames,costs,color=colors,edgecolor='black')

plt.title('Avg Cost for two in Well Known Coffee Shops in New Delhi',fontsize=12)

plt.ylabel('Average Cost')

plt.xlabel('Coffee Shops')

plt.show()



#bar graph to plot average ratings of Well known Coffee Shops

ratings = delhi_coffee_shops['aggregate rating']

plt.barh(rnames,ratings,color=colors,edgecolor='black')

plt.title('Average Ratings for Popular Coffee Shops in New Delhi',fontsize=12)

plt.xlabel('Average Rating')

plt.ylabel('Coffee Shops')

plt.show()
delhi_local_coffee_shops = rest_new_delhi[(rest_new_delhi['cuisines']=='Cafe') 

                                          & (~rest_new_delhi['restaurant name'].isin(coffee_shops))]



top_coffee_shops=delhi_local_coffee_shops.groupby('restaurant name',as_index=False)[['average cost for two','aggregate rating']].mean().round(2).sort_values(ascending=False,by='aggregate rating')



top_coffee_shops=top_coffee_shops[top_coffee_shops['aggregate rating'] >= 4].sort_values(ascending=False,by='aggregate rating')

#plot top local coffee with average rating more than 4

rnames = top_coffee_shops['restaurant name']

colors = ['#B00303','#DC0508','#DC0508','#F9908D','#F7CAC9','#F7CAC9']

ratings = top_coffee_shops['aggregate rating']

plt.bar(rnames,ratings,color=colors,edgecolor='black')

plt.title('Top Local Coffee Shops with min 4 rating in New Delhi',fontsize=12)

plt.ylabel('Average Rating')

plt.xlabel('Coffee Shops')

plt.xticks(rotation='vertical')

plt.show()

expensive_coffee_shops=delhi_local_coffee_shops.groupby('restaurant name',as_index=False)[['average cost for two','aggregate rating']].mean().round(2).sort_values(ascending=False,by='average cost for two').head(6)



#plot 6 expensive local coffee shops

rnames = expensive_coffee_shops['restaurant name']

colors = ['#B00303','#B00303','#DC0508','#F9908D','#F7CAC9','#F7CAC9']

ratings = expensive_coffee_shops['average cost for two']

plt.bar(rnames,ratings,color=colors,edgecolor='black')

plt.title('Expensive Local Coffee Shops in New Delhi',fontsize=12)

plt.ylabel('Average Cost for two')

plt.xlabel('Coffee Shops')

plt.xticks(rotation='vertical')

plt.show()
correlation = delhi_local_coffee_shops.groupby('restaurant name',as_index=False)[['average cost for two','aggregate rating']].mean().round(2).sort_values(ascending=False,by='aggregate rating')



#plot scatter graph to analyse correlation

weight = correlation['aggregate rating']

height = correlation['average cost for two']

plt.figure(figsize=(10,8))

plt.scatter(weight,height,c='g',marker='o')

plt.xlabel('Average Rating')

plt.ylabel('Average Cost')

plt.title('Average Rating Vs Average Cost for Local Coffee Shops')

plt.show()