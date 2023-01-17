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
songs = pd.read_csv('/kaggle/input/top-spotify-songs-from-20102019-by-year/top10s.csv', encoding='ISO-8859-1', index_col=0)

songs.columns.rename('column_name').to_frame().reset_index(drop=True)
songs = songs.rename(columns={'top genre':'genre', 'nrgy':'energy', 'dnce':'dance', 'dB':'db', 'dur':'length', 'acous':'acoustic', 'spch':'speech'}).reset_index(drop=True)

songs.index += 1

songs.head()
songs.shape
songs[['pop','title','artist','year']].sort_values(by='pop', ascending=False).iloc[:10].set_index('pop')
songs.groupby('artist').size().sort_values(ascending=False).rename('no_of_songs').head(10).to_frame()
temp = songs.groupby('artist').size().idxmax()

songs[songs.artist == temp].set_index('pop').sort_values(by='pop',ascending=False)
songs[['bpm', 'energy', 'dance', 'db', 'live', 'val', 'length', 'acoustic', 'speech', 'pop']].groupby('pop').mean().iloc[-5:].sort_values(by='pop', ascending=False)
songs.groupby('genre').pop.count().rename('number').sort_values(ascending=False).head(5).to_frame().plot.pie(subplots=True)
songs[songs.genre == 'dance pop'].sort_values(by='pop', ascending = False).set_index('pop').head(10)
songs.set_index('pop').sort_values(by='pop',ascending=False).iloc[:100].groupby('year')[['bpm', 'energy', 'dance', 'db', 'live', 'val', 'length', 'acoustic', 'speech']].mean().round(2).sort_values(by='year',ascending=False)
songs[songs.genre == 'dance pop'].sort_values(by=['pop'],ascending=False)[['bpm', 'energy', 'dance', 'db', 'live', 'val', 'length', 'acoustic', 'speech']].head(100).mean().rename('mean').to_frame().plot.barh()