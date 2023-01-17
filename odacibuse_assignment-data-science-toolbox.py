import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import json  # will use for converting json data to str



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# First load data

data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

# look data inside

data.head()

data.info()
highestBudgetMovies = data.nlargest(10, 'budget')  # we keep 10 movies which has highest budget

plt.subplots(figsize=(12, 10))

ax = pd.Series(highestBudgetMovies.budget).sort_values(ascending=True).plot.barh(width=0.9,

                                                                                 color=sns.color_palette('rocket', 10))

for i, v in enumerate(highestBudgetMovies.title):

    ax.text(.8, i, v, fontsize=12, color='white', weight='bold')

plt.title('Highest budget movies')

plt.xlabel("Total budget in $")

ax.set_yticklabels([])  # hide movie's id

plt.show()

sns.relplot(kind='scatter', x='budget', y='popularity', data=data)

plt.show()

# lets look genres

print(data.genres)

# JSON to STRING

# json.loads; used for parse valid JSON String into Python dictionary.

# range; allows to generate a series of numbers within a given range. Ex; 0,1,2...10

# append; adds a single item to the existing list

data['genres'] = data['genres'].apply(json.loads)  # with apply function each of data.genres will become string

for index, i in zip(data.index, data['genres']):

    list1 = []

    for j in range(len(i)):

        list1.append((i[j]['name']))  # 'name' key contains name of the genre



    data.loc[index, 'genres'] = str(list1)

# STRING to LIST

# The strip; removes characters from both left and right based on the arg.

data['genres'] = data['genres'].str.strip('[]').str.replace(' ', '').str.replace("'", '')

data['genres'] = data['genres'].str.split(',')

plt.subplots(figsize=(12, 10))

list1 = []

# extend(); takes a single argument (a list) and adds it to the end.

# value_counts(); return a series containing counts of unique values.

for i in data.genres:

    list1.extend(i)  # we add all genres to one list

ax = pd.Series(list1).value_counts()[:10].sort_values(ascending=True).plot.barh(width=0.9,

                                                                                color=sns.color_palette('spring', 10))



# We take each value of unique genres and indexing them like (0:2297),(1:1722) etc and write them to bar.

for i, v in enumerate(pd.Series(list1).value_counts()[:10].sort_values(ascending=True).values):

    ax.text(.8, i, v, fontsize=12, color='white', weight='bold')

plt.title('Top 10 Genres')

plt.show()

data['nice'] = ['good' if each < data.vote_average.mean() else 'not' for each in data.vote_average]

# We add new column to our dataset which has 2 unique value

data.nice.unique()

#  Pie chart, where the slices will be ordered and plotted counter-clockwise.

labels = 'Good', 'Not Good'

sizes = data['nice'].value_counts()  # sizes have number for 'not' and 'good'

explode = (0.05, 0)  # first part of pie will be explode

colors = ["lightpink", "yellowgreen"]

plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True)  # autopct; wrote rates on pies

plt.title('Good movie or not?')

plt.legend(labels, loc='upper right')

plt.show()

data['time'] = ['short' if each < data.runtime.mean() else 'not' for each in data.runtime]

# We add new column to our dataset which has 2 unique value

data.time.unique()

labels = 'Short', 'Not Short'

sizes = data['time'].value_counts()  # sizes have number for 'not' and 'short'

explode = (0.05, 0)  # first part of pie will be explode

colors = ["bisque", "aqua"]

plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True)  # autopct; wrote rates on pies

plt.title('Short movie or not?')

plt.legend(labels, loc='upper right')

plt.show()

sns.relplot(kind='line', x='vote_average', y='runtime', data=data)

plt.show()
