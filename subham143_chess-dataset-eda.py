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

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
games = pd.read_csv("../input/chess/games.csv")
games.columns
games.isnull().sum()
games.head(2)
plt.figure(figsize = (12,8))

ax = games.winner.value_counts()[:25].plot(kind = 'bar')

ax.legend(['* WIN'])

plt.xlabel("Winer color")

plt.ylabel("Count of  win")

plt.title("Winer color vs Number of win",fontsize =20, weight = 'bold')
plt.figure(figsize = (12,8))

ax = games.victory_status.value_counts()[:25].plot(kind = 'bar')

ax.legend(['* victory_status'])

plt.xlabel("type of victory status")

plt.ylabel("Count of  victory status")

plt.title("victory status vs Number of victory status",fontsize =20, weight = 'bold')
plt.style.use('dark_background')

sns.distplot(games.turns,color='white')

plt.show()
plt.style.use('default')

sns.jointplot(x='white_rating',y='black_rating',data=games,kind='hex',color='black')
plt.style.use('default')

colors = ["white", "darkgrey","red"]

customPalette = sns.set_palette(sns.color_palette(colors))

sns.violinplot(x="winner", y="turns", data=games,palette=customPalette)
plt.style.use('grayscale')



sns.regplot(x='black_rating',y='turns',data=games,scatter_kws={'s':2})
# rated games vs non rated games length

plt.style.use('default')

sns.countplot(games.rated)
sns.countplot(y="black_id", data=games, palette="Reds",

              order=games.black_id.value_counts().iloc[:10].index)
plt.style.use('default')

plt.figure(figsize=(30, 10))

sns.countplot(games.opening_name,order=games.opening_name.value_counts().iloc[:8].index)
games.victory_status.value_counts()
games = games[games.rated]  # only rated games

games['mean_rating'] = (games.white_rating + games.black_rating) / 2

games['rating_diff'] = abs(games.white_rating - games.black_rating)
under_1500 = games[games.mean_rating < 1500]

under_2000 = games[games.mean_rating < 2000]

over_2000 = games[games.mean_rating > 2000]



brackets = [under_1500, under_2000, over_2000]

bracket_titles = ['Under 1500', 'Under 2000', 'Over 2000']
plt.figure(figsize=(15,11))

for i, bracket in enumerate(brackets):

    victory_status = bracket.victory_status.value_counts()

    plt.subplot(1, 4, i+1)

    plt.title(bracket_titles[i])

    plt.pie(victory_status, labels=victory_status.index)
mate_games = games[games.victory_status=='mate']



under_1500 = mate_games[mate_games.mean_rating < 1500]

under_2000 = mate_games[mate_games.mean_rating < 2000]

over_2000 = mate_games[mate_games.mean_rating > 2000]



m_brackets = [under_1500, under_2000, over_2000]
turn_means = [b.turns.mean() for b in m_brackets]



plt.figure(figsize=(10,5))

plt.ylim(0, 100)

plt.title('Number of turns until mate')

plt.plot(bracket_titles, turn_means, 'o-', color='r')
plt.figure(figsize=(10,5))

plt.scatter(mate_games.mean_rating, mate_games.turns)
white_upsets = games[(games.winner == 'white') & (games.white_rating < games.black_rating)]

black_upsets = games[(games.winner == 'black') & (games.black_rating < games.white_rating)]

upsets = pd.concat([white_upsets, black_upsets])
THRESHOLD = 900

STEP = 50



u_percentages = []



print(f'Rating difference : Percentage of wins by weaker player')

for i in range(0+STEP, THRESHOLD, STEP):

    th_upsets = upsets[upsets.rating_diff > i]

    th_games = games[games.rating_diff > i]

    upsets_percentage = (th_upsets.shape[0] / th_games.shape[0]) * 100

    u_percentages.append([i, upsets_percentage])

    print(f'{str(i).ljust(18)}:  {upsets_percentage:.2f}%')
plt.figure(figsize=(10,5))

plt.plot(*zip(*u_percentages))

plt.xlabel('rating difference')

plt.ylabel('upsets percentage')