# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Loading the dataset 'movies_metadata.csv'

md = pd.read_csv('../input/movies_metadata.csv')

md.head()
md['genres'].head()
# Finding the number of null values from the dataset

md.isnull().sum()
# Converting all the genres names from a dictionary into a list

from ast import literal_eval

md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x : [i['name'] for i in x] if isinstance(x, list) else [])
md['genres'].head()
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')

vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')

# Calculating the mean of vote_averages

C = vote_averages.mean()

C
# The minimum votes required to be listed in the chart. We will use 95th percentile as our cutoff.

# i.e. For a movie to feature in the charts, it must have more votes than at least 95% of the movies in the list.

m = vote_counts.quantile(0.95)

m
md['year'] = pd.to_datetime(md['release_date'], errors = 'coerce').apply(lambda x : str(x).split('-')[0] if x != np.nan else np.nan)
# Movies that qualify our criterion

qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

qualified['vote_count'] = qualified['vote_count'].astype('int')

qualified['vote_average'] = qualified['vote_average'].astype('int')

qualified.shape
# Defining the weighted rating function

def weighted_rating(x):

    v = x['vote_count']

    R = x['vote_average']

    return (v/(v+m) * R) + (m/(m+v) * C)
# Creating the weighed rating column

qualified['wr'] = qualified.apply(weighted_rating, axis = 1)
# Sorting the values in the qualified data frame according to weighted rating values

qualified = qualified.sort_values('wr', ascending = False)

qualified.head(10)
# Defining a function to build charts for particular genres

s = md.apply(lambda x: pd.Series(x['genres']), axis = 1).stack().reset_index(level = 1, drop = True)

s.name = 'genre'

gen_md = md.drop('genres', axis = 1).join(s)
def build_chart(genre, percentile = 0.80):

    df = gen_md[gen_md['genre'] == genre]

    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')

    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')

    C = vote_averages.mean()

    m = vote_counts.quantile(percentile)

    qualified = df[(df['vote_count'] >= m ) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]

    qualified['vote_count'] = qualified['vote_count'].astype('int')

    qualified['vote_average'] = qualified['vote_average'].astype('int')

    

    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis = 1)

    

    qualified = qualified.sort_values('wr', ascending = False)

    

    return qualified
build_chart('Action', 0.80).head(15)