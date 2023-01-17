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
potw = pd.read_csv('../input/NBA_player_of_the_week.csv')
potw.head()
potw.tail()
potw.describe()
potw.apply(lambda x: x.count(), axis=0)
potw_no_conference = potw.dropna(how = 'any')
potw_no_conference.apply(lambda x : x.count(), axis=0)
potw_no_conference['Age'].hist()
potw['Age'].hist()
## here we use the groupby function
player_by_year = potw['Player'].groupby(potw['Draft Year'])
player_by_year.count()
player_by_year.count().plot(title="The Number of Players Won POTW according to Draft Year")

player_by_team = potw['Player'].groupby(potw['Team'])
player_by_team.count().plot.pie(title="POTW by Teams", figsize=(10,10))
player_by_position = potw['Player'].groupby(potw['Position'])
player_by_position.count().plot.pie(title="POTW by Positions", figsize=(10,10))
player_by_name = potw['Player'].groupby(potw['Player'])
player_by_name.count()
player = pd.DataFrame(player_by_name.count())
player.sort_values(by='Player', ascending=False)
