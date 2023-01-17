# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
nba_players = pd.read_csv("../input/NBA_player_of_the_week.csv")
nba_players.head()
positions=nba_players.groupby('Position')[['Height','Seasons in league']] #get player positions and time in league
positions.groups
positions.get_group('PG') #get point guards players of the week
nba_players.loc[nba_players["Seasons in league"]<2] #rookies who are players of the week
nba_players