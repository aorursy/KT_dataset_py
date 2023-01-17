# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/tv-shows-on-netflix-prime-video-hulu-and-disney/tv_shows.csv', index_col=0)

df.head()
df.info()
df_RT = df.drop(['IMDb', 'Age', 'type'], axis=1)

df_RT = df_RT.dropna().reset_index(drop=True)
df_RT.sort_values('Rotten Tomatoes').reset_index(drop=True).head(10)
df_IM = df.drop(['Rotten Tomatoes', 'Age', 'type'], axis=1)

df_IM = df_IM.dropna().reset_index(drop=True)
df_IM.sort_values('IMDb', ascending=False).reset_index(drop=True).head(10)
l = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

IMDB_by_platform = pd.Series([df_IM[df_IM[x]==1]['IMDb'].mean() for x in l], l)

IMDB_by_platform
IMDB_by_year = df_IM.groupby('Year')['IMDb'].agg(['mean', 'count']).sort_values('mean', ascending=False)

IMDB_by_year.columns = ['IMDb Mean', 'Number of Shows']

IMDB_by_year.head(10)
fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(IMDB_by_year.sort_index()['IMDb Mean'])

ax.set_xlabel('Year')

ax.set_ylabel('IMDb Mean')

fig.show()
df_temp = IMDB_by_year.sort_index()



fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(df_temp.index, df_temp['IMDb Mean'] * IMDB_by_year.sort_index()['Number of Shows'], c='r', linestyle='dashed')

ax.plot(df_temp.index, df_temp['Number of Shows'], c='b')

ax.set_xlabel('Year')

ax.legend(df_temp.columns, loc='best', fontsize=20)

fig.show()