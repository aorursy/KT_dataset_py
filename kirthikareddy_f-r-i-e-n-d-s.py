# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
frnds=pd.read_csv('/kaggle/input/friends-series-dataset/friends_episodes_v2.csv')

frnds.head()
frnds.info()
frnds.describe()
season_dur = frnds.groupby('Season').Duration.sum().to_frame().reset_index()

plt.figure(figsize=(10,4))

sns.barplot(x=season_dur.Season, y=season_dur.Duration, palette='inferno')

plt.title("duration of each season",fontsize=15)

plt.xlabel('Duration',fontsize=14)

plt.ylabel('Season',fontsize=14)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

plt.ylim(200,600)
season_eps = frnds.groupby('Season').count().Year_of_prod.to_frame()

season_eps.columns = ['Episode Number']

season_eps
season_stars = frnds.groupby('Season').mean().Stars.to_frame().reset_index()

season_stars.columns = ['Season','Average Stars']

season_stars = season_stars.sort_values('Average Stars', ascending=False)



plt.figure(figsize=(10,5))

sns.barplot(y=season_stars.Season, x=season_stars['Average Stars'], palette='inferno', orient='h')

plt.title('Average IMDB Stars of each Season', fontsize=15)

plt.xlabel('Average Stars', fontsize=13)

plt.ylabel('Season', fontsize=13)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.xlim(8.2,8.8)

pop_eps10 = frnds[['Episode_Title','Stars']].sort_values('Stars', ascending=False).head(10).reset_index(drop=True)

pop_eps10.Episode_Title[2] = 'The Last One: Part 2'

pop_eps10
np.arange(8, 9, 0.1)
plt.figure(figsize=(10,5))

sns.barplot(y=pop_eps10.Episode_Title, x=pop_eps10.Stars, palette='Blues_d')

plt.title('Top 10 High-rated Episodes', fontsize=15)

plt.xlabel('IMDB Stars', fontsize=13)

plt.ylabel('Episode', fontsize=13)

plt.xticks(np.arange(9, 9.8, 0.1), fontsize=12)

plt.yticks(fontsize=12)

plt.xlim(9, 9.8)

active_dirs = frnds.Director.value_counts().to_frame().reset_index()

active_dirs.columns = ['Director', 'Number of Episodes']

dir_stars_counts = frnds.groupby('Director').Stars.sum().to_frame().sort_values('Stars', ascending=False).reset_index()

print(active_dirs)
dirs = dir_stars_counts.merge(active_dirs, how='left', on='Director')

dirs['Average'] = round(dirs.Stars / dirs['Number of Episodes'], 2)

dirs_plt = dirs[dirs['Number of Episodes']>=10].sort_values('Average', ascending=False)

print(dirs_plt)
plt.figure(figsize=(10,5))

sns.barplot(y=dirs_plt.Director, x=dirs_plt.Average, palette='Oranges_d')

plt.title('Top 8 High-rated Directors (Directing more than 10 Episodes)', fontsize=15)

plt.xlabel('Average IMDB Stars of the Episodes', fontsize=13)

plt.ylabel('Director', fontsize=13)

plt.xticks(np.arange(8, 8.8, 0.1), fontsize=12)

plt.yticks(fontsize=12)

plt.xlim(8, 8.7)
