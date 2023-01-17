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
game_data =  pd.read_csv("../input/nfl-big-data-bowl-2021/games.csv")
player_data = pd.read_csv("../input/nfl-big-data-bowl-2021/players.csv")
play_data = pd.read_csv("../input/nfl-big-data-bowl-2021/plays.csv")
week_data = pd.DataFrame()
for i in range(1,18):
    week = "week"+str(i)+".csv"
    df = pd.read_csv("../input/nfl-big-data-bowl-2021/"+week)
    week_data = pd.concat([df],ignore_index = True)
week_data = week_data.filter(['nflId','gameId','playId', 's', 'a'])
play_data = play_data.filter(['gameId', 'playId', 'passResult', 'penaltyCodes', 'playResult'])
play_data.merge(week_data, left_on='gameId', right_on = 'gameId')
play_data
play_data = play_data.merge(week_data)
play_data = play_data[play_data.nflId.notnull()]
test = play_data.merge(player_data)
play_data.loc[play_data['gameId'] == 2018122200]
