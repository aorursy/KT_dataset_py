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
cups = pd.read_csv('../input/WorldCups.csv')
matches = pd.read_csv('../input/WorldCupMatches.csv')
players = pd.read_csv('../input/WorldCupPlayers.csv')
cups.head()
matches.head()
players.tail()
# matches.columns # for check column name
home_game = matches[matches['Home Team Name'] == 'Korea Republic']
away_game = matches[matches['Away Team Name'] == 'Korea Republic']
home_game.head()
away_game.tail()
game_win
game_lose