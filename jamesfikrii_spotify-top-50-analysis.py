# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')
df = pd.read_csv('/kaggle/input/top50spotify2019/top50.csv',encoding='ISO-8859-1')
df.head()
df.rename({

    'Unnamed: 0' : 'rank',

    'Track.Name' : 'track_name',

    'Artist.Name' : 'artist_name',

    'Genre' : 'genre',

    'Beats.Per.Minute' : 'beats_per_minute',

    'Energy' : 'energy',

    'Danceability' : 'danceability',

    'Loudness..dB..' : 'loudness_db',

    'Liveness' : 'liveness',

    'Valence.' : 'valence',

    'Length.' : 'length',

    'Acousticness..' : 'acousticness',

    'Speechiness.' : 'speechiness',

    'Popularity' : 'popularity'

},axis=1,inplace=True)
df['genre'].value_counts().head()
df['artist_name'].value_counts().head(10)
df.genre.value_counts()
df_group_by_genre = df.groupby('genre').mean()
def plot_genre_analysis(feat):    

    plt.figure(figsize=(8,6))

    sns.barplot(data=df,y=df_group_by_genre.sort_values(by=feat,ascending=False).index,x=df_group_by_genre.sort_values(by=feat,ascending=False)[feat])
for col in df.select_dtypes(exclude='O').columns:

    plot_genre_analysis(col)
for item in df.select_dtypes(exclude='O').drop('rank',axis=1):

    fig,ax = plt.subplots(nrows=1,ncols=1)

    sns.distplot(df[item],ax=ax)

for item in df.select_dtypes(exclude='O').drop('rank',axis=1):

    the_most = df[df[item] == df[item].max()][['rank','track_name','artist_name','genre',item]]

    print('Song with the highest {} = "{}" rank {} with {} = {}'.format(item,the_most['track_name'].values[0],the_most['rank'].values[0],item,the_most[item].values[0]) )

    print('----')
column_to_pair = df.select_dtypes(exclude='O').drop('popularity',axis=1).columns

for item in column_to_pair:

    fig,ax = plt.subplots(nrows=1,ncols=1)

    sns.scatterplot(df['popularity'],df[item],ax=ax)
column_to_pair = df.select_dtypes(exclude='O').drop('rank',axis=1).columns

for item in column_to_pair:

    fig,ax = plt.subplots(nrows=1,ncols=1)

    sns.scatterplot(df['rank'],df[item],ax=ax)