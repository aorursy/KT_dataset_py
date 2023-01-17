# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# to search missing (nan) values
import missingno as msno

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_players = pd.read_csv('../input/WorldCupPlayers.csv')
data_cup = pd.read_csv('../input/WorldCups.csv')
data_players
data_players_somecolumns = data_players.loc[:50, ['Coach Name', 'Player Name', 'Position', 'Event']]
data_players_somecolumns
msno.matrix(data_players_somecolumns)
plt.show()
msno.matrix(data_players)  # all data
msno.bar(data_players_somecolumns)
plt.show()