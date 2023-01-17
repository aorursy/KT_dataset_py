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
#import libraries separate from the one kaggle already gave
from dateutil.relativedelta import relativedelta
import datetime as dt
#g is games, p is for plays, and plyrs is for player data
g = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/games.csv')
p = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/plays.csv')
plyrs = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2021/players.csv')
#compile weekly game data into a single data frame
file_name = '/kaggle/input/nfl-big-data-bowl-2021/week{}.csv'
game_list = []
for i in range(1,18):
    game_list.append(pd.read_csv(file_name.format(i)))

df = pd.concat(game_list)
df.head()
#convert relevant datetime variables
df['time'] = pd.to_datetime(df.time, format='%Y-%m-%dT%H:%M:%S.%f%z')
plyrs['birthDate'] = pd.to_datetime(plyrs['birthDate'],format= '%Y-%m-%d',infer_datetime_format=True)
g['gameDate'] = pd.to_datetime(g.gameDate, format = '%Y-%m-%d',infer_datetime_format=True)
g['gameTimeEastern'] = pd.to_datetime(g.gameDate, format = '%H:%M:%S',infer_datetime_format=True)
#create a current date variable to calculate age for players
plyrs['current_date'] = dt.date.today()
plyrs['current_date'] = pd.to_datetime(plyrs.current_date,format= '%Y-%m-%d')
plyrs.head()
#create age variable in years
plyrs['age'] = plyrs.current_date - plyrs.birthDate
plyrs['age'] = plyrs.age / np.timedelta64(1,'Y')
plyrs.head()
p.head()
p.isnull().sum()
g.isnull().sum()
plyrs.isnull().sum()
df.isnull().sum()
df.head()
