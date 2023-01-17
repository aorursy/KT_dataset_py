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

all_seasons = pd.read_csv("../input/nba-players-data/all_seasons.csv")

all_seasons.head()
all_seasons.shape

all_seasons.loc[all_seasons.player_name == 'LeBron James', ['player_name', 'pts', 'season']]
all_seasons.query("country == 'Spain'")
all_seasons.loc[all_seasons.player_name == 'Chris Paul'].ast.mean()
grouped = all_seasons.groupby(['player_name', 'season'])

grouped.first()
seasons_played = all_seasons.player_name.value_counts()

seasons_played.head()
import matplotlib.pyplot as plt



lebron_statistics = all_seasons.loc[all_seasons.player_name == 'LeBron James', ['pts','season']]

plt.plot(lebron_statistics["season"], lebron_statistics["pts"])

plt.ylabel('Points Per Game')

plt.xticks(rotation=90)

plt.title("Lebron James Average Points Per Season")

plt.xlabel('Season')

plt.show()
all_seasons['pts'].sort_values(ascending=False).head(10)
all_seasons2 = all_seasons.loc[all_seasons.player_name == 'Kobe Bryant']

all_seasons2['pts'].max()