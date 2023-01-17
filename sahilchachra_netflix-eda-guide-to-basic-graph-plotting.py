import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



%matplotlib inline
# Importing Data set

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/netflix-shows/netflix_titles.csv')

dataCopy = data

data.head(10)
#Checking columns names

list(data.columns)
data.info()
data.shape[0] # number of entries
sns.set(style='darkgrid')

sns.countplot(x = 'type',data=data,palette='Set1')
plt.figure(figsize=(12,10))

plt.title('Top 15 Countries on the basis of Content Creation')

sns.countplot(data=data,y='country',order=data['country'].value_counts().index[0:15],palette='Accent')
plt.figure(figsize=(12,10))

sns.set(style='darkgrid')

ax = sns.countplot(y='release_year',data=data,order=data['release_year'].value_counts().index[0:15],palette='Set1')
movie = data[data['type']=='Movie']

#movie.columns

movie.head(5)
duration = []

movie = movie[movie['duration'].notna()]

for i in movie['duration']:

    duration.append(int(i.strip('min')))
plt.figure(1,figsize=(15,10))

plt.title("Duration of Movies")

sns.distplot(duration)
plt.figure(figsize=(15,8))

plt.title('Directors with Most movies')

sns.countplot(y='director',data=movie,order=movie['director'].value_counts().index[0:10],palette='Set3')
genrePerMovie=[]

totalMoveGenre = []

setGenre = set()

set1 = set()

for i in movie['listed_in']:

    if(type(i)==str):

        g = i.split(',')

        for genre in g:

            setGenre.add(genre.strip())
totalMovieGenre = list(setGenre)

#len(totalMovieGenre)
%%time

storeCountOfGenre = {}

currentGenre = []

for actualGenre in totalMovieGenre:

    count = 1

    for i in movie['listed_in']:

        currentGenre = []

        if(type(i)==str):

            s=i.split(',')

            for j in s:

                currentGenre.append(j.strip())

            if(actualGenre in currentGenre):

                if actualGenre not in storeCountOfGenre:

                    storeCountOfGenre[actualGenre] = 1

                else:

                    storeCountOfGenre[actualGenre] +=1
import operator

import itertools



sorted_Genre = dict(sorted(storeCountOfGenre.items(), key=operator.itemgetter(1),reverse=True))

finalSortedListOfGenre = dict(itertools.islice(sorted_Genre.items(),10))



keysGenre = list(finalSortedListOfGenre.keys())

keysGenre = keysGenre[1:]

valuesGenre = list(finalSortedListOfGenre.values())

valuesGenre = valuesGenre[1:]



#[1:] is done because after all this calculations, 'Internal Movies' came up in the top as most movie had this Genre.

#But 'International Movie' is not a genre. So a temporary solution is to remove 1st element from value and key list
import matplotlib.cm as cm

from matplotlib.colors import Normalize

from numpy.random import rand

dataColorGenre = [2, 3, 5, 6, 8, 12, 7, 5,9,11,10,4]

my_cmap = cm.get_cmap('ocean')

my_norm = Normalize(vmin=0, vmax=8)



plt.figure(figsize=(10,7))

plt.bar(keysGenre, valuesGenre, color=my_cmap(my_norm(dataColorGenre)))

plt.xticks(rotation=90)

plt.show()
castPerMovie=[]

totalCast = []

set1 = set()

for i in movie['cast']:

    if(type(i)==str):

        s = i.split(',')

        for j in s:

            set1.add(j.strip())
# Run this to check if any cast has repeated ( CROSS VERIFY IF THE ABOVE CODE WORKS AS EXPECTED)

#from collections import Counter

#Counter(l)
totalCast = list(set1)

#len(totalCast)
%%time

storeCounts = {}

currentCasts = []

for actualCast in totalCast:

    count = 1

    for i in movie['cast']:

        currentCasts = []

        if(type(i)==str):

            s=i.split(',')

            for j in s:

                currentCasts.append(j.strip())

            if(actualCast in currentCasts):

                if actualCast not in storeCounts:

                    storeCounts[actualCast] = 1

                else:

                    storeCounts[actualCast] +=1
sorted_d = dict(sorted(storeCounts.items(), key=operator.itemgetter(1),reverse=True))

finalSortedList = dict(itertools.islice(sorted_d.items(),20))



keys = finalSortedList.keys()

values = finalSortedList.values()
plt.figure(figsize=(10,5))



dataColorDirector = [2, 3, 5, 6, 8, 12, 7, 5,9,11,10,4]

my_cmap = cm.get_cmap('rainbow')

my_norm = Normalize(vmin=0, vmax=8)



plt.bar(keys,values,color=my_cmap(my_norm(dataColorDirector)))

plt.xticks(rotation=90)
data = dataCopy

series = data[data['type']=='TV Show']

series.head(5)
durationSeries = []

tvshow = series[series['duration'].notna()]

for i in tvshow['duration']:

    durationSeries.append(int(i.strip('Season')))



plt.figure(figsize=(12,10))

plt.title('Average no. of Seasons of TV Shows')

sns.distplot(durationSeries)
setCountry = set()

for country in series['country']:

    if(type(country) == str):

        s = country.split(',')

        for singleCountry in s:

            setCountry.add(singleCountry.strip())
#setCountry

totalCountriesForSeries = list(setCountry)

#len(totalCountriesForSeries)
%%time

currentCountries = []

countCountryTvShows = dict()

for singleCountry in totalCountriesForSeries:

    currentCountries = []

    for country in series['country']:

        if(type(country)==str):

            s = country.split(',')

            for j in s:

                currentCountries.append(j.strip())

            if(singleCountry in currentCountries):

                if(singleCountry not in countCountryTvShows):

                    countCountryTvShows[singleCountry] = 1

                else:

                    countCountryTvShows[singleCountry]+=1
sorted_countCountriesTvShows = dict(sorted(countCountryTvShows.items(), key=operator.itemgetter(1),reverse=True))

finalSortedDictTvShows = dict(itertools.islice(sorted_countCountriesTvShows.items(),10))



keys = finalSortedDictTvShows.keys()

values = finalSortedDictTvShows.values()
#finalSortedDictTvShows



plt.figure(figsize=(10,5))



dataColorDirector = [1, 2, 3, 4, 5, 6, 7, 8,9,10,11,12]

my_cmap = cm.get_cmap('jet')

my_norm = Normalize(vmin=0, vmax=8)



plt.bar(keys,values,color=my_cmap(my_norm(dataColorDirector)))

plt.xticks(rotation=90)
%%time

castPerShow=[]

totalCastTvShow = []

tvShowset = set()

for i in series['cast']:

    if(type(i)==str):

        s = i.split(',')

        for j in s:

            tvShowset.add(j.strip())

            

totalCastTvShow = list(tvShowset)

len(totalCastTvShow)
%%time

storeCountCastTvShow = {}

currentCastsTvShow = []

for actualTvShowCast in totalCastTvShow:

    for i in series['cast']:

        currentCastsTvShow = []

        if(type(i)==str):

            s=i.split(',')

            for j in s:

                currentCastsTvShow.append(j.strip())

            if(actualTvShowCast in currentCastsTvShow):

                if actualTvShowCast not in storeCountCastTvShow:

                    storeCountCastTvShow[actualTvShowCast] = 1

                else:

                    storeCountCastTvShow[actualTvShowCast] +=1

                    

sortedCastTvShow = dict(sorted(storeCountCastTvShow.items(), key=operator.itemgetter(1),reverse=True))

finalSortedListTvShowCast = dict(itertools.islice(sortedCastTvShow.items(),10))



keys = finalSortedListTvShowCast.keys()

values = finalSortedListTvShowCast.values()
plt.figure(figsize=(10,5))



dataColorDirector = [12, 11, 10, 9, 8, 7, 6, 5,4,3,2,1]

my_cmap = cm.get_cmap('flag')

my_norm = Normalize(vmin=0, vmax=8)



plt.bar(keys,values,color=my_cmap(my_norm(dataColorDirector)))

plt.title("Top 10 Casts with no. of Shows")

plt.xticks(rotation=90)
genrePerShow=[]

totalGenreShow = []

setGenreShow = set()

setShowGenre = set()

for i in series['listed_in']:

    if(type(i)==str):

        g = i.split(',')

        for genre in g:

            setShowGenre.add(genre.strip())

            

totalShowGenre = list(setShowGenre)

len(totalShowGenre)
%%time

storeCountOfShowGenre = {}

currentShowGenre = []

for actualShowGenre in totalShowGenre:

    for i in series['listed_in']:

        currentShowGenre = []

        if(type(i)==str):

            s=i.split(',')

            for j in s:

                currentShowGenre.append(j.strip())

            if(actualShowGenre in currentShowGenre):

                if actualShowGenre not in storeCountOfShowGenre:

                    storeCountOfShowGenre[actualShowGenre] = 1

                else:

                    storeCountOfShowGenre[actualShowGenre] +=1
sortedShowGenre = dict(sorted(storeCountOfShowGenre.items(), key=operator.itemgetter(1),reverse=True))

finalSortedListOfShowGenre = dict(itertools.islice(sortedShowGenre.items(),11))



keysShowGenre = list(finalSortedListOfShowGenre.keys())

keysShowGenre = keysShowGenre[1:]



valuesShowGenre = list(finalSortedListOfShowGenre.values())

valuesShowGenre = valuesShowGenre[1:]



#print(keysShowGenre,valuesShowGenre)

#[1:] is done because after all this calculations, 'Internal Movies' came up in the top as most movie had this Genre.

#But 'International Movie' is not a genre. So a temporary solution is to remove 1st element from value and key list
dataColorGenre = [2, 3, 5, 6, 8, 12, 7, 5,9,11,10,4]

my_cmap = cm.get_cmap('ocean')

my_norm = Normalize(vmin=0, vmax=8)



plt.figure(figsize=(10,7))

plt.bar(keysShowGenre, valuesShowGenre, color=my_cmap(my_norm(dataColorGenre)))

plt.title("Top 10 Genre in TV Shows")

plt.xticks(rotation=90)

plt.show()