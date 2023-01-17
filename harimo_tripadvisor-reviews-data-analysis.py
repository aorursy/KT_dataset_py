# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import ast

import operator

from matplotlib import cm

from itertools import cycle, islice

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
reviews = pd.read_csv('../input/TA_restaurants_curated.csv')
reviews.head()
reviews.describe()
reviews.info()
reviews.columns
reviews.drop(['Unnamed: 0', 'URL_TA', 'ID_TA', 'Ranking'], axis = 1, inplace=True)
reviews.head()
reviews.count()
reviews['City'].nunique()
plt.figure(figsize=(10,7), dpi =100)

plot = sns.countplot(reviews['City'], order=reviews['City'].value_counts().index)

plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)

plt.tight_layout()
byCity = reviews.groupby('City')

byCity['Rating'].mean()
x = list()

y = list()

for city in list(reviews['City'].unique()):

    x.append(city)

    y.append(reviews[reviews['City'] == city]['Rating'].mean())

fig, ax = plt.subplots(1,1,figsize=(17,7))

ax.bar(x,y,color = 'cyan',edgecolor = 'black')

ax.set_ylim(bottom=3.5)

ax.set_xticklabels(labels = x, rotation = 45)

ax.set_xlabel('City')

ax.set_ylabel('Average Review Rating')
x = list()

y = list()

for city in list(reviews['City'].unique()):

    x.append(city)

    y.append(reviews[reviews['City'] == city]['Rating'].max())

fig, ax = plt.subplots(1,1,figsize=(17,7))

ax.bar(x,y,color = 'cyan',edgecolor = 'black')

ax.set_ylim(bottom=3.5)

ax.set_xticklabels(labels = x, rotation = 45)

ax.set_xlabel('City')

ax.set_ylabel('Max Review Rating')
### Min Review rating per city
x = list()

y = list()

for city in list(reviews['City'].unique()):

    x.append(city)

    y.append(reviews[reviews['City'] == city]['Rating'].min())

fig, ax = plt.subplots(1,1,figsize=(17,7))

ax.bar(x,y,color = 'cyan',edgecolor = 'black')

ax.set_xticklabels(labels = x, rotation = 45)

ax.set_xlabel('City')

ax.set_ylabel('Min Review Rating')
print('Total negative ratings count : ', len(reviews[reviews['Rating'] < 0]))

reviews[reviews['Rating'] < 0]
x = list()

y = list()

count = 0

for city in list(reviews['City'].unique()):

    count = len(reviews[(reviews['City'] == city) & (reviews['Rating'] < 0)])

    if count > 0:

        y.append(count)

        x.append(city)

    count = 0

fig, ax = plt.subplots(1,1,figsize=(17,7))

ax.bar(x,y,color = 'cyan',edgecolor = 'black')

ax.set_xticklabels(labels = x, rotation = 45)

ax.set_xlabel('City')

ax.set_ylabel('Count of Negative Ratings')
city = list(reviews['City'].unique())

fig, axes = plt.subplots(nrows=7,ncols=5,figsize=(17,20))

i = 0

ratings = list(reviews['Rating'].unique())

ratingsCount = list()

for c in city:

    reviewsCity = reviews[reviews['City'] == c]

    plot = sns.countplot(x='Rating', data = reviewsCity, ax=axes.flatten()[i])

    plot.set_title(c)

    plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)

    plt.tight_layout()

    i = i + 1    
reviews.head(4)
reviews.isna().sum()
reviewsCity.head()
fig, axes = plt.subplots(6,2, figsize=(17,15))

columns = ['Cuisine Style', 'Rating', 'Price Range', 'Number of Reviews', 'Reviews']

i = 0

y_ax = list()

y_ax1 = list()

x_ax = list(reviews['City'].unique())

for col in columns:

    for city in list(reviews['City'].unique()):

        na = reviews[reviews['City'] == city][col].isna().sum()

        notna = reviews[reviews['City'] == city][col].notna().sum()

        y_ax.append(na)

        y_ax1.append(notna)

        #print('Percentage of NaN per City per Column : ', (na*100/(na+notna)), city, col)

    axes[i][0].bar(x_ax, y_ax)

    axes[i][1].bar(x_ax, y_ax1)

    y_ax = list()

    y_ax1 = list()

    for k in range(0,2):

        axes[i][k].set_xticklabels(x_ax, rotation = 90)

        axes[i][k].set_ylabel(col)

    plt.tight_layout()

    i = i+1
from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize

import matplotlib.cm

from numpy import meshgrid
##### place = list(reviews['City'].unique())

data = {'City': ['Amsterdam', 'Athens', 'Barcelona', 'Berlin', 'Bratislava', 'Brussels', 'Budapest', 'Copenhagen', 'Dublin', 'Edinburgh', 'Geneva', 'Hamburg', 'Helsinki', 'Krakow', 'Lisbon', 'Ljubljana', 'London', 'Luxembourg', 'Lyon', 'Madrid', 'Milan', 'Munich', 'Oporto', 'Oslo', 'Paris', 'Prague', 'Rome', 'Stockholm', 'Vienna', 'Warsaw', 'Zurich'],

        'Lat':  [52.38, 37.98, 42.38, 52.52, 48.14, 50.85, 47.49, 55.67, 53.34, 55.95, 46.20, 55.55, 60.16, 50.06, 38.72, 46.05, 51.50, 49.81, 45.76, 40.41, 45.46, 48.13, 41.15, 59.91, 48.85, 50.07, 41.90, 59.32, 48.20, 52.22, 47.37],

        'Long': [4.9,   23.72, 2.17,  13.40, 17.10, 4.35,  19.04, 12.56, -6.26, -3.18,  6.14,  9.99, 24.93, 19.94, -9.13, 15.50, 0.12,  6.12,  4.83,  -3.70,  9.19, 11.58, -8.62, 10.75,  2.35, 14.43, 12.49, 18.06, 16.37, 21.01, 8.54]}



dfr = pd.DataFrame(data, columns = ['City', 'Lat', 'Long'])



#print(place)

print(data['City'])

newDf = pd.merge(reviews, dfr, how='left', on='City')

newDf.head()
plt.figure(figsize=(10,10))

map = Basemap(projection='aeqd', lon_0 = 10, lat_0 = 50, width = 5000000, height = 5000000, resolution='l') # set res=h

map.drawmapboundary(fill_color='cyan')

map.etopo()

map.drawcoastlines()

map.drawcountries()

map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

scale = 0.00002

for c in list(newDf['City'].unique()):

    #print(c, ":", newDf[newDf['City'] == c]['Number of Reviews'].sum())

    dfr.loc[dfr['City'] == c, 'Total Num of Reviews'] = newDf[newDf['City'] == c]['Number of Reviews'].sum()

for i in range(0,len(dfr)):

    x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])

    map.plot(x,y,marker='o', color='Red', markersize=10)

plt.show()
plt.figure(figsize=(10,10))

map = Basemap(projection='aeqd', lon_0 = 10, lat_0 = 50, width = 5000000, height = 5000000, resolution='l') # set res=h

map.drawmapboundary(fill_color='cyan')

map.etopo()

map.drawcoastlines()

map.drawcountries()

map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

#map.set_cmap('hot')

scale = 0.00002

for c in list(newDf['City'].unique()):

    #print(c, ":", newDf[newDf['City'] == c]['Number of Reviews'].sum())

    dfr.loc[dfr['City'] == c, 'Total Num of Reviews'] = newDf[newDf['City'] == c]['Number of Reviews'].sum()

for i in range(0,len(dfr)):

    x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])

    map.plot(x,y,marker='o', color='Red', alpha = 0.6, markersize=int(dfr.ix[i,'Total Num of Reviews']*scale))

plt.show()
for r in list(newDf['Rating'].unique()):

    for c in list(newDf['City'].unique()):

        #print(c, ":", newDf[newDf['City'] == c]['Number of Reviews'].sum())

        dfr.loc[dfr['City'] == c, r] = len(newDf[(newDf['City'] == c) & (newDf['Rating'] == r)])

dfr.drop(np.nan, axis = 1, inplace = True)
for r in [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0]:

    if r in [5.0, 4.5, 4.0, 3.5, 3.0]:

        scale = 0.01

    elif r in [2.5, 2.0, 1.5, 1.0, -1.0]:

        scale = 0.1

    plt.figure(figsize=(10,10))

    plt.title("{} Star Rating".format(r))

    map = Basemap(projection='aeqd', lon_0 = 10, lat_0 = 50, width = 5000000, height = 5000000, resolution='l') # set res=h

    map.drawmapboundary(fill_color='cyan')

    map.etopo()

    map.drawcoastlines()

    map.drawcountries()

    map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

    

    for i in range(0,len(dfr)):

        x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])

        map.plot(x,y,marker='o', color='Red', alpha = 0.6, markersize=int(dfr.ix[i,r]*scale))

    plt.show()
reviews.isna().sum()
reviews[reviews['City'] == 'London']['Rating'].isna().sum()
byCity = reviews.groupby('City')

byCity['Rating'].mean()
reviews['Rating'].fillna(value = 4.0, inplace = True)

reviews['Rating'].isna().sum()
for r in list(newDf['Rating'].unique()):

    for c in list(newDf['City'].unique()):

        #print(c, ":", newDf[newDf['City'] == c]['Number of Reviews'].sum())

        dfr.loc[dfr['City'] == c, r] = len(newDf[(newDf['City'] == c) & (newDf['Rating'] == r)])

dfr.drop(np.nan, axis = 1, inplace = True)



scale = 0.01

r = 4.0

plt.figure(figsize=(10,10))

plt.title("{} Star Rating".format(r))

map = Basemap(projection='aeqd', lon_0 = 10, lat_0 = 50, width = 5000000, height = 5000000, resolution='l') # set res=h

map.drawmapboundary(fill_color='cyan')

map.etopo()

map.drawcoastlines()

map.drawcountries()

map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

for i in range(0,len(dfr)):

    x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])

    map.plot(x,y,marker='o', color='Red', alpha = 0.6, markersize=int(dfr.ix[i,r]*scale))

plt.show()
for city in list(reviews['City'].unique()):

    print(city, reviews[reviews['City'] == city]['Price Range'].isna().sum())
fig, axes = plt.subplots(1,1, figsize=(7,5))

y_ax = list()

x_ax = list(reviews['City'].unique())

for city in list(reviews['City'].unique()):

    na = reviews[reviews['City'] == city]['Price Range'].isna().sum()

    y_ax.append(na)  

axes.bar(x_ax, y_ax)

axes.set_xticklabels(x_ax, rotation = 90)

axes.set_ylabel('{} NA count'.format(col))

axes.set_xlabel('City')

plt.tight_layout()
reviews['Price Range'].unique()
reviews['City'].count()
for i in range(0, reviews['City'].count()):

    if reviews.loc[i,'Price Range'] == '$':

        reviews.loc[i,'Price'] = 1

    elif reviews.loc[i,'Price Range'] == "$$ - $$$":

        reviews.loc[i,'Price'] = 2

    elif reviews.loc[i,'Price Range'] == '$$$$':

        reviews.loc[i,'Price'] = 3

reviews['Price'][np.isnan(reviews['Price'])] = 0
print(reviews['Price'].nunique())

print(reviews['Price'].unique())

sns.countplot(x='Price', data=reviews)
sns.countplot(x='Rating', data = reviews, hue = 'Price')
city = list(reviews['City'].unique())

fig, axes = plt.subplots(nrows=7,ncols=5,figsize=(17,20))

i = 0

ratings = list(reviews['Rating'].unique())

ratingsCount = list()

for c in city:

    reviewsCity = reviews[reviews['City'] == c]

    plot = sns.countplot(x='Price', data = reviewsCity, ax=axes.flatten()[i])

    plot.set_title(c)

    plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)

    plt.tight_layout()

    i = i + 1  
for p in list(reviews['Price'].unique()):

    for c in list(reviews['City'].unique()):

        dfr.loc[dfr['City'] == c, str('{} Dollar'.format(p))] = len(reviews[(reviews['City'] == c) & (reviews['Price'] == p)])

for p in ['3.0 Dollar', '2.0 Dollar', '1.0 Dollar', '0.0 Dollar']:

    if p in ['3.0 Dollar']:

        scale = 0.05

    elif p in ['1.0 Dollar']:

        scale = 0.01

    elif p in ['2.0 Dollar', '0.0 Dollar']:

        scale = 0.005

    plt.figure(figsize=(10,10))

    plt.title("{} Price".format(p))

    map = Basemap(projection='aeqd', lon_0 = 10, lat_0 = 50, width = 5000000, height = 5000000, resolution='l') # set res=h

    map.drawmapboundary(fill_color='cyan')

    map.etopo()

    map.drawcoastlines()

    map.drawcountries()

    map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

    

    for i in range(0,len(dfr)):

        x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])

        map.plot(x,y,marker='o', color='Red', alpha = 0.6, markersize=int(dfr.ix[i,p]*scale))

    plt.show()
for i in range(0, reviews['City'].count()):

    if reviews.loc[i,'Price'] == 0:

        reviews.loc[i,'Price'] = 2
print(reviews['Price'].nunique())

print(reviews['Price'].unique())

sns.countplot(x='Price', data=reviews)
sns.countplot(x='Rating', data = reviews, hue = 'Price')
for p in list(reviews['Price'].unique()):

    for c in list(reviews['City'].unique()):

        dfr.loc[dfr['City'] == c, str('{} Dollar'.format(p))] = len(reviews[(reviews['City'] == c) & (reviews['Price'] == p)])
for p in ['3.0 Dollar', '2.0 Dollar', '1.0 Dollar']:

    if p in ['3.0 Dollar']:

        scale = 0.05

    elif p in ['1.0 Dollar']:

        scale = 0.01

    elif p in ['2.0 Dollar']:

        scale = 0.005

    plt.figure(figsize=(10,10))

    plt.title("{} Price".format(p))

    map = Basemap(projection='aeqd', lon_0 = 10, lat_0 = 50, width = 5000000, height = 5000000, resolution='l') # set res=h

    map.drawmapboundary(fill_color='cyan')

    map.etopo()

    map.drawcoastlines()

    map.drawcountries()

    map.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

    

    for i in range(0,len(dfr)):

        x, y = map(dfr.ix[i,'Long'], dfr.ix[i,'Lat'])

        map.plot(x,y,marker='o', color='Red', alpha = 0.6, markersize=int(dfr.ix[i,p]*scale))

    plt.show()
city = list(reviews['City'].unique())

fig, axes = plt.subplots(nrows=7,ncols=5,figsize=(17,20))

i = 0

ratings = list(reviews['Rating'].unique())

ratingsCount = list()

for c in city:

    reviewsCity = reviews[reviews['City'] == c]

    plot = sns.countplot(x='Price', data = reviewsCity, ax=axes.flatten()[i])

    plot.set_title(c)

    plot.set_xticklabels(plot.get_xticklabels(), rotation = 45)

    plt.tight_layout()

    i = i + 1   
#Counting function to parse the cuisine lists

def cuisine_count(_list):

    cuisine_dict = {'UnknownCuisine': 0}

    for cuisines in _list:

        if cuisines is not np.nan:

            cuisines = ast.literal_eval(cuisines)  

            for cuisine in cuisines:  

                if cuisine in cuisine_dict:

                    cuisine_dict[cuisine] += 1

                else :

                    cuisine_dict[cuisine] = 1

        else:

            cuisine_dict['UnknownCuisine'] +=1

    #print(cuisines)

    return(cuisine_dict)
def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:d}".format(absolute)

def plotCuisine(cui,count,type = 'bar',city=""):

    if type == 'bar':

        fig, axes = plt.subplots(1,1, figsize = (20,20))

    elif type == 'pie':    

        fig, axes = plt.subplots(1,1, figsize = (7,7))

    lists = sorted(cui.items(), key=operator.itemgetter(1),reverse=True)

    x, y = zip(*lists[:count])

    col = cm.inferno_r(np.linspace(.7,.2, len(x)))

    if type == 'bar':

        axes.bar(x,y,color = col)

        axes.set_ylabel('Count')

        axes.set_xlabel('Cuisine')

        axes.set_xticklabels(x,rotation = 90)

    elif type == 'pie':

        axes.pie(y, labels = x, autopct=lambda pct: func(pct, y))

        axes.set_title(city,fontsize=15)

    axes.set_facecolor('lightgrey')
cui = cuisine_count(reviews['Cuisine Style'])

plotCuisine(cui, len(cui))
for city in list(reviews['City'].unique()):

    cui = cuisine_count(reviews[reviews['City'] == city]['Cuisine Style'])

    plotCuisine(cui,10,'pie',city)
reviews['Number of Reviews'].isna().sum()
x = list()

y = list()

for city in list(reviews['City'].unique()):

    x.append(city)

    y.append(reviews[reviews['City'] == city]['Number of Reviews'].sum())

fig, ax = plt.subplots(1,1,figsize=(17,7))

ax.bar(x,y,color = 'cyan',edgecolor = 'black')

ax.set_ylim(bottom=3.5)

ax.set_xticklabels(labels = x, rotation = 45)

ax.set_xlabel('City')

ax.set_ylabel('Average Review Rating')
for city in list(reviews['City'].unique()):

    print('----------',city,'----------', '\n',reviews[(reviews['City'] == city) & (reviews['Rating'] > 4.0)].sort_values(by = 'Number of Reviews', axis = 0, ascending = False)[['Name', 'Number of Reviews']].head())

    print("\n")