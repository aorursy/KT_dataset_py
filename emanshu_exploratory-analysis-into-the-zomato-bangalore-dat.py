%matplotlib inline

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns ; sns.set()

import os

# print(os.listdir("../input"))

import plotly

plotly.__version__

import plotly.plotly as py

import plotly.graph_objs as go
rawData = pd.read_csv('../input/zomato.csv')

rawData.head()
print('Data types')

print(rawData.dtypes)

print('Info')

print(rawData.info())

print('Rows and Columns')

print(rawData.shape)
print(rawData.columns)

rawData.drop(['url','phone',],axis=1,inplace=True)

rawData = rawData.rename( columns={'approx_cost(for two people)':'cost_for_two','listed_in(type)':'restaurant_type','listed_in(city)':'city'} )

rawData = rawData.replace('SantÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ© Spa Cuisine','Spa Cuisine')
# Calculate percentage of restaurants with no rating. 

newRest = (rawData['rate'] == 'NEW').sum()

totalRatings= (rawData['rate'].count())

nanValues = (rawData['rate'].isnull()).sum()

percentOfNoRatings = ((nanValues + newRest)  / totalRatings * 100).round(2)

print (f'There are {totalRatings} restaurants in the Zomato data set that have ratings. Out of the total 51,717 restaurants {percentOfNoRatings}% are new or have no rating ')
ratingCount = pd.DataFrame(rawData.groupby('rate')['rate'].count())

# drop na values

dataFiltered = rawData.dropna()

# filter out ratings that have a value of NEW

dataFiltered = dataFiltered[dataFiltered['rate'] != 'NEW']



# reduce rating column to just rating

dataFiltered['rate'] = dataFiltered['rate'].astype(str)

dataFiltered['rate'] = dataFiltered['rate'].apply(lambda rate: rate[:3])

dataFiltered['rate'] = dataFiltered['rate'].astype(float)



# remove ',' from cost for two column

dataFiltered['cost_for_two'] = dataFiltered['cost_for_two'].apply(lambda x: x.replace(',','')).astype(int)
sns.set_style('white')

fig, ax = plt.subplots(figsize=(15,9))

sns.distplot(dataFiltered['rate'],bins=15,color="m",kde=False,)

plt.title('Distribution of Ratings')

plt.xlabel('Rating')

sns.despine()
sns.set_style('white')

fig, ax = plt.subplots(figsize=(30,9))

sns.distplot(dataFiltered['cost_for_two'],color="m",kde=False,)

plt.title('Distribution of Meal Cost')

plt.xlabel('Cost for Two')

plt.xticks(dataFiltered['cost_for_two'].unique(),rotation='vertical')

plt.xlim(40,3000)

sns.despine()
# Need to remove , from cost column // need to move this above meal cost dist

# dataFiltered['cost_for_two'] = dataFiltered['cost_for_two'].apply(lambda x: x.replace(',','')).astype(int)

# plotting

fig, ax = plt.subplots(figsize=(30,9))

ax = sns.scatterplot(y='rate',x='cost_for_two',data=dataFiltered, color='m',palette='plasma')

ax.set_title('Cost vs Ratings')

ax.set_ylabel('Ratings')

ax.set_xlabel('Cost for Two')

sns.set_style('white')

sns.despine()
plt.figure(figsize=(20,5))



sns.regplot(x='rate',y='votes',data=dataFiltered,color='m')

plt.ylim(0,17500)

plt.xlabel('Rating')

plt.ylabel('Votes')

plt.title('Rating vs Votes')

sns.despine()
# create a column that has the number of cusines served

dataFiltered['cuisines_count'] = dataFiltered['cuisines'].apply(lambda x: len(x.split(','))).astype(int)

#plot it



plt.figure(figsize=(10,5))

sns.set_style('white')

sns.scatterplot(x='cuisines_count', y='votes',data=dataFiltered,palette='plasma')

sns.despine()

plt.ylabel('Votes')

plt.xlabel('Cuisines Served')
topTenRestTypesByVotes = pd.DataFrame(dataFiltered.groupby(['rest_type'])[['votes']].sum()).sort_values(by='votes',ascending=False).astype(int)

topTenRestTypesByVotes.reset_index(inplace=True)



plt.figure(figsize=(15,5))

sns.barplot(topTenRestTypesByVotes['votes'].iloc[:10],topTenRestTypesByVotes['rest_type'].iloc[:10],palette='plasma')

sns.despine()

plt.ylabel('Restaurant Type')

plt.xlabel('Total Votes')

plt.title('Top Ten Restaurant Types by Total Votes')
topTenTypeByRate = pd.DataFrame(dataFiltered.groupby(['rest_type','name'])['rate'].mean()).sort_values(by='rate',ascending=False).iloc[:10].reset_index()

topTenTypeByRate



plt.figure(figsize=(15,5))

sns.barplot(topTenTypeByRate['rate'].iloc[:10],topTenTypeByRate['rest_type'].iloc[:10],palette='plasma')

sns.despine()

plt.ylabel('Restaurant Type')

plt.xlabel('Average Rating')

plt.title('Top Ten Restaurant Types by Average Rating')
topTenByVotes = pd.DataFrame(dataFiltered.groupby(['name','rest_type'])['votes'].sum()).sort_values(by='votes',ascending=False).iloc[0:10].reset_index()



plt.figure(figsize=(15,5))

sns.barplot(topTenByVotes['votes'].iloc[:10],topTenByVotes['name'].iloc[:10],palette='plasma')

sns.despine()

plt.ylabel('Restaurant')

plt.xlabel('Total Votes')

plt.title('Top Ten Restaurants by Total Votes')
topTenByRatings = pd.DataFrame(dataFiltered.groupby(['name'])['rate'].mean()).sort_values(by='rate',ascending=False).reset_index()



fig, axs = plt.subplots(figsize=(10,5),)

sns.barplot(topTenByRatings['rate'].iloc[:10],topTenByRatings['name'].iloc[:10],palette='plasma')

sns.despine()

plt.ylabel('Restaurant')

plt.xlabel('Average Rating')

plt.title('Top Ten Restaurants by Average Rating')

plt.xlim(3.5,5.0)
highestRatedByLocation = pd.DataFrame(dataFiltered.groupby(['city'])['rate'].mean()).sort_values(by='rate',ascending=False).reset_index()

highestRatedByLocation



sns.set_style('white')



plt.figure(figsize=(5,5))

sns.barplot(y=highestRatedByLocation['city'].iloc[:10],x=highestRatedByLocation['rate'].iloc[:10],palette='plasma')

plt.title('Location of Highest Rated Restaurants')

plt.ylabel('City')

plt.xlabel('Average Rating')

plt.xlim(3.5,5.0)

sns.despine()

cityTotalVotes = pd.DataFrame(dataFiltered.groupby('city')['votes'].sum()).sort_values(by='votes',ascending=False).reset_index()

cityTotalVotes



sns.set_style('white')

plt.figure(figsize=(10,5))

sns.barplot(y=cityTotalVotes['city'].iloc[:10],x=cityTotalVotes['votes'].iloc[:10],palette='plasma')

plt.title('Location of Highest Rated Restaurants')

plt.ylabel('City')

plt.xlabel('Average Rating')

# plt.xlim(3.5,5.0)

sns.despine()
# f, (ax1, ax2) = plt.subplots(2)

# sns.regplot(x, y, ax=ax1)

# sns.kdeplot(x, ax=ax2)



# f, (ax1, ax2) = plt.subplots(1,2,figsize=(20,5))

# sns.barplot(y=cityTotalVotes['city'].iloc[:10],x=cityTotalVotes['votes'].iloc[:10],palette='plasma', ax=ax1)



# plt.title('Location of Highest Rated Restaurants')

# plt.ylabel('City')

# plt.xlabel('Average Rating')

# plt.xlim(3.5,5.0)



# sns.barplot(y=highestRatedByLocation['city'].iloc[:10],x=highestRatedByLocation['rate'].iloc[:10],palette='plasma', ax=ax2)



# f[0] = plt.title('Location of Highest Rated Restaurants')

# ax2 = plt.ylabel('City')

# ax2 = plt.xlabel('Average Rating')



# f.tight_layout()