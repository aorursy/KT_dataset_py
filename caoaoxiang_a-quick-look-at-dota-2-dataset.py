

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
matches = pd.read_csv('../input/match.csv', index_col=0)

players = pd.read_csv('../input/players.csv')

player_time = pd.read_csv('../input/player_time.csv')



hero_names = pd.read_csv('../input/hero_names.csv')



chat = pd.read_csv('../input/chat.csv')

objectives = pd.read_csv('../input/objectives.csv')



teamfights_players = pd.read_csv('../input/teamfights_players.csv')

teamfights = pd.read_csv('../input/teamfights.csv')
matches.head()
players.iloc[:10,:15]
players['account_id'].value_counts()
players.iloc[:5,:]
players.iloc[:5,40:55]
player_time.head()
a_match = player_time.query('match_id == 1')
a_match.T
teamfights.head()
teamfights_players.head()
chat.head()