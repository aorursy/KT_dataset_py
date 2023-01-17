# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

from collections import Counter

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv')
data = data.iloc[:,1:]
# Each row represents a single track
data.head()
data.dtypes
pop = data.sort_values('popularity', ascending=False).head(20)

pop = pop[['artists', 'name', 'popularity', 'year']]
pop
# The most popular songs are all recent musics (2018 -)

# These songs are playing a lot, as they are recent people tend to play them more

# They are in the Top 50 Songs or Musics of the Moment which is why they are more played
# Most popular / played not from last 10 years recent years

pop2 = data.sort_values('popularity', ascending=False)

pop2 = pop2[pop2.year < 2010]

pop2 = pop2[['artists', 'name', 'popularity', 'year']]

pop2 = pop2.head(20)
pop2
pop_year = data.sort_values('popularity', ascending=False).groupby('year').first()

pop_year = pop_year.reset_index()
pop_year = pop_year[['year', 'artists', 'name', 'popularity']]
pop_year.head(50)
pop_year.tail(50)
ax = pop_year.plot.bar(x='year', y='popularity', figsize=(20,10))
# Pick in 1933

pop_year[pop_year.year == 1933]
data[data['name'].str.contains("All of Me")].sort_values('popularity', ascending = False).head(5)[['year', 'artists', 'name', 'popularity']]
# Maybe people search for All of Me from John Legend, which can explain the popularity of this 1933 song
def splitDataFrameList(df,target_column,delimiters):

    ''' df = dataframe to split,

    target_column = the column containing the values to split

    separator = the symbol used to perform the split

    returns: a dataframe with each entry for the target column separated, with each element moved into a new row. 

    The values in the other columns are duplicated across the newly divided rows.

    '''

    regexPattern = "|".join(map(re.escape,delimiters))

    def splitListToRows(row,row_accumulator,target_column,regexPattern):

        split_row = re.split(regexPattern,row[target_column])

        for s in split_row:

            new_row = row.to_dict()

            new_row[target_column] = s

            row_accumulator.append(new_row)

    new_rows = []

    df.apply(splitListToRows,axis=1,args = (new_rows,target_column,regexPattern))

    new_df = pd.DataFrame(new_rows)

    return new_df
# Including feat artists

feat_artists =  splitDataFrameList(data,'artists',',')

feat_artists['artists'] = feat_artists['artists'].map(lambda x: re.sub(r'\W+', '', x))
feat_artists.head()
# At least 50 songs

feat_count = feat_artists.groupby('artists').count().iloc[:,0]

feat_count = feat_count.reset_index()

feat_count.columns = ['artists', 'count']

feat_artists = pd.merge(feat_artists, feat_count, on='artists')
feat_artists['count'] = feat_artists['count'].astype(int)
feat_artists = feat_artists[feat_artists['count'] >= 50]
feat_pop = feat_artists.groupby('artists')['popularity'].mean()

feat_pop = feat_pop.reset_index()

feat_pop = feat_pop.sort_values('popularity', ascending=False)
feat_pop.head(20)
from ast import literal_eval

all_artists = data.copy()

all_artists['artists'] = all_artists['artists'].map(lambda x: literal_eval(x))

all_artists['artists'] = all_artists['artists'].map(lambda x: x[0])
artists_count = all_artists.groupby('artists').count().iloc[:,0]

artists_count = artists_count.reset_index()

artists_count.columns = ['artists', 'count']

all_artists = pd.merge(all_artists, artists_count, on='artists')
all_artists['count'] = all_artists['count'].astype(int)
all_artists = all_artists[all_artists['count'] >= 50]
artists_pop = all_artists.groupby('artists')['popularity'].mean()

artists_pop = artists_pop.reset_index()

artists_pop = artists_pop.sort_values('popularity', ascending=False)
artists_pop.head(20)
danceability = data.groupby('year')['danceability'].mean()

danceability = danceability.reset_index()

danceability.columns = ['year', 'mean']
ax = danceability.plot.bar(x='year', y='mean', figsize=(20,10))
# Decrease until early 50s then increase
speechiness = data.groupby('year')['speechiness'].mean()

speechiness = speechiness.reset_index()

speechiness.columns = ['year', 'mean']
ax = speechiness.plot.bar(x='year', y='mean', figsize=(20,10))
# More speechiness before the 50s then increasing from 1990 (maybe rap songs ?)
instrumentalness = data.groupby('year')['instrumentalness'].mean()

instrumentalness = instrumentalness.reset_index()

instrumentalness.columns = ['year', 'mean']
ax = instrumentalness.plot.bar(x='year', y='mean', figsize=(20,10))
# Huge decrease from 1950

# In 1933, songs with more acapela ?
explicit = data.groupby('year')['explicit'].mean()

explicit = explicit.reset_index()

explicit.columns = ['year', 'mean']
ax = explicit.plot.bar(x='year', y='mean', figsize=(20,10))
# More explicit contents since the 90s
data[(data.explicit == 1) & (data.year == 1921)]
# There might be some noise for the year 1921
data_w_genres = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv')

data_by_genres = pd.read_csv('/kaggle/input/spotify-dataset-19212020-160k-tracks/data_by_genres.csv')
data_w_genres = data_w_genres[['artists', 'genres']]
# Taking the first artists, not including feat which lead to biais song genres

genres = pd.merge(all_artists, data_w_genres, on='artists')
multiple_genres = splitDataFrameList(genres,'genres',',')

multiple_genres['genres'] = multiple_genres['genres'].map(lambda x: re.sub(r'\W+', '', x))
# Filter genres with no values

multiple_genres = multiple_genres[multiple_genres.genres != '']
genres_year = multiple_genres.groupby('year')['genres'].apply(list)

genres_year = genres_year.reset_index()
genres_year['most_common'] = genres_year['genres'].map(lambda x: Counter(x).most_common(1)[0][0])

genres_year['count_mc'] = genres_year['genres'].map(lambda x: Counter(x).most_common(1)[0][1])                                        
total_year = all_artists.groupby('year').count().iloc[:,1]

total_year = total_year.reset_index()

total_year.columns = ['year', 'total']
genres_year = pd.merge(genres_year, total_year, on='year')
genres_year['perc_mc'] = genres_year['count_mc'] / genres_year['total']
fig,ax = plt.subplots()

fig.set_size_inches(20,10)

ax = sns.barplot(x='year',y='perc_mc',hue='most_common', data=genres_year, dodge=False, palette='Paired')

plt.xticks(rotation=90)

ax.legend(loc='upper right')

plt.show()
# From classic to traditional pop music to rock then rap