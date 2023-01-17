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
imdb_movies  = pd.read_csv('../input/imdb-movies.csv')

movie_count = imdb_movies.shape[0]

columns_to_keep = [

    'popularity',

    'budget',

    'revenue',

    'original_title',

    'director',

    'runtime',

    'genres',

    'production_companies',

    'release_year',    

]

imdb_movies = imdb_movies[columns_to_keep]

animated_movies = imdb_movies[imdb_movies['genres'].str.contains('Animation') == True]

animated_movies_count = animated_movies.shape[0]

animated_movies = animated_movies.assign(profit=pd.Series(animated_movies.revenue - animated_movies.budget).values)

animated_movies.head(5)
print('Percentage of animated movies: {}%'.format(round((animated_movies_count/movie_count) * 100, 2)))
animated_movies['release_year'].value_counts(sort=False).plot.bar(figsize=(20,5))
animated_movies['production_companies'].value_counts().head(5).plot.bar(figsize=(10,5))
animated_movies.loc[animated_movies['popularity'].idxmax(), 'original_title']
animated_movies.loc[animated_movies['budget'].idxmax(), 'original_title']
animated_movies.loc[animated_movies['profit'].idxmax(), 'original_title']
top_10 = animated_movies.nlargest(10, 'profit')

top_10.index = top_10.original_title

top_10[['original_title','profit']].plot.bar(figsize=(12,6))
plt.figure(figsize=(12,6))

plt.title("Profit Vs Year")

plt.xlabel('Year')

plt.ylabel('Profit in Billions')

plt.scatter(animated_movies.release_year, animated_movies.profit, color='red')

plt.show()