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

import matplotlib.pyplot as plt 

%config InlineBackend.figure_format ='retina'

%matplotlib inline



data = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='ISO-8859-1', index_col = 0)

print(data.head())

# Songs

print(data[data['year'] == 2011][['title', 'artist']])
avgs = data.groupby('year').mean()

print(avgs.head())

column_names = avgs.columns

for column in column_names:

    plt.figure()

    plt.plot(avgs.index, avgs[column])

    plt.title(column)
# Find similar songs

print(data.iloc[94])



def find_similar_songs(data, song_prop):

    df = data

    properties = ['bpm','nrgy','dnce','val','acous','spch']

    # Dataframe filters - bpm, nrgy, dnce, val, acous, spch

    for prop in properties:

        df = df[(abs(df[prop] - song_prop[prop]) < 15)]

    return df 
# Get song

my_song = data.iloc[94]

sim_songs = find_similar_songs(data, my_song)

print(sim_songs)