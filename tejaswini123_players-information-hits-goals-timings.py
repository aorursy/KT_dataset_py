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
import pandas as pd
game=pd.read_csv("../input/game.csv")
game.head(10)
import pandas as pd
players=pd.read_csv('../input/player_info.csv')
players.head(10)
import pandas as pd
game_states=pd.read_csv('../input/game_teams_stats.csv')
game_states.head(10)
import pandas as pd
game_info=pd.read_csv('../input/game_skater_stats.csv')
game_info.head(10)
#joining 3 dataframes
df = pd.merge(pd.merge(players,game_info ,on='player_id'),game_states, on = 'game_id') 
df.head(5)
#players-Max no.of goals
print(df[['firstName','goals_x']][df.goals_x == df.goals_x.max()])
#players-Max no.of goals
print(df[['firstName','hits_x']][df.hits_x == df.hits_x.max()])
#no.of players, goals is 0
df[df.goals_x==0].count()["player_id"]
#import pandas as pd
#df.sort_values("player_id", inplace = True)
#playstore.sort_values("Rating", inplace = True)
#filter = df["goals_x"] == 0
#filter2 = df.player_id.value_counts()
#df.where(filter , inplace = True)

import pandas as pd
game_info=pd.read_csv('../input/game_shifts.csv')
game_info.head(10)
game_shifts=pd.read_csv("../input/game_shifts.csv")
game_shifts.head(5)
game_shifts['seconds_played'] = game_shifts['shift_end'] - game_shifts['shift_start']
df_time_played = game_shifts.groupby(['game_id','player_id'])['seconds_played'].sum().reset_index()
df_time_played['minutes_played'] = df_time_played['seconds_played'] / 60
df_time_played['hours_played'] = df_time_played['minutes_played'] / 60 
df_time_played.head()
#joining 2 data frames game_shifts and players
df_player_time= pd.merge(players,df_time_played, on=['player_id','player_id'])
#players - max time played.
print(df_player_time[['player_id','firstName','minutes_played']][df_player_time.minutes_played == df_player_time.minutes_played.max()])
#players- minimum time played
print(df_player_time[['player_id','firstName','minutes_played']][df_player_time.minutes_played == df_player_time.minutes_played.min()])
#how many players participated from each team.
df.nationality.value_counts()
#players-Max no.of hits
print(df[['firstName','hits_x']][df.hits_x == df.hits_x.max()])
#players-how many times particited in each game
games=df.game_id
df.firstName.value_counts()
