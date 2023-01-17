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
games = pd.read_csv("../input/nfl-big-data-bowl-2021/games.csv")

games
games.info()
# let's take a sample date to gain an inference

sample_date = "09/09/2018"

games[games['gameDate']==sample_date]
per_date_games = games['gameDate'].value_counts().reset_index()

per_date_games.columns = ['date', 'count']





plt.rcdefaults()

fig, ax = plt.subplots(figsize=(16, 10))





ax.barh(np.arange(len(per_date_games)), per_date_games['count'].values, align='center')

ax.set_yticks(np.arange(len(per_date_games)))

ax.set_yticklabels(per_date_games['date'].tolist())

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Games count --->')

ax.set_title('Number of games on every date --->')



plt.show()
per_date_games['count'].value_counts().plot.pie()

plt.show()
per_time_games = games['gameTimeEastern'].value_counts().reset_index()

per_time_games.columns = ['Time', 'count']





plt.rcdefaults()

fig, ax = plt.subplots(figsize=(16, 10))





ax.barh(np.arange(len(per_time_games)), per_time_games['count'].values, align='center')

ax.set_yticks(np.arange(len(per_time_games)))

ax.set_yticklabels(per_time_games['Time'].tolist())

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Times count --->')

ax.set_title('Number of games on every duration/Time --->')



plt.show()
players = pd.read_csv("../input/nfl-big-data-bowl-2021/players.csv")

players
players.info()
players['weight'].hist(bins=25)

plt.show()
plays = pd.read_csv("../input/nfl-big-data-bowl-2021/plays.csv")

plays
plays.info()
is_nan_plays = pd.DataFrame()

is_nan_plays['Attrs'] = plays.isna().sum().index

is_nan_plays['count'] = plays.isna().sum().values

is_nan_plays = is_nan_plays.sort_values(by=['count'], ascending=False).reset_index(drop=True)



plt.rcdefaults()

fig, ax = plt.subplots(figsize=(16, 10))





ax.barh(np.arange(len(is_nan_plays)), is_nan_plays['count'].values, align='center')

ax.set_yticks(np.arange(len(is_nan_plays)))

ax.set_yticklabels(is_nan_plays['Attrs'].tolist())

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Times count --->')

ax.set_title('Number of games on every duration/Time --->')



plt.show()