import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from subprocess import check_output

plt.style.use('fivethirtyeight')

print(check_output(["ls", "../input"]).decode("utf8"))
players = pd.read_csv('../input/nba-players-stats-20142015/players_stats.csv')

shots = pd.read_csv('../input/nba-shot-logs/shot_logs.csv')
players['Name'] = players.Name.apply(lambda x: x.strip(',.').lower())

players['Average_Dist'] = players.Name.apply(lambda x: np.mean(shots.SHOT_DIST[shots.player_name == x]))

age_df = pd.DataFrame()

age_df['Age'] = np.unique(players.Age)

age_df['Dist'] = age_df.Age.apply(lambda x: 0.304*np.mean(players.Average_Dist[(players.Age == x) & ((players.Pos == 'SG') | (players.Pos == 'PG'))]))



age_df = age_df[~(age_df.Age == 36)]

plt.plot(age_df.Age,age_df.Dist,'o')

x = age_df['Age'][0:19]

y = age_df['Dist'][0:19]



fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit)

plt.plot(x,fit_fn(x),'r')



age_df = age_df[~(age_df.Age == 36)]

x = age_df['Age'][0:18]

y = age_df['Dist'][0:18]



fit = np.polyfit(x,y,1)

fit_fn = np.poly1d(fit)

plt.plot(x,fit_fn(x),'r--')





plt.legend(['Data','Fit','Excluding age 39'])



plt.title('Average shot distance vs Age')

plt.xlabel('Age')

plt.ylabel('Average Shot Distance [m]')
players['2PA'] = players.FGA - players['3PA']

age_df['2PA'] = age_df.Age.apply(lambda x: np.sum(players['2PA'][players.Age == x]))

age_df['3PA'] = age_df.Age.apply(lambda x: np.sum(players['3PA'][players.Age == x]))



age_df['3P ratio'] = age_df['3PA']/(age_df['2PA'] + age_df['3PA'])

age_df['2P ratio'] = 1 - age_df['3P ratio']



age_df['2PA_guard'] = age_df.Age.apply(lambda x: np.sum(players['2PA'][(players.Age == x) & ((players.Pos == 'PS') | (players.Pos.values == 'SG'))]))

age_df['3PA_guard'] = age_df.Age.apply(lambda x: np.sum(players['3PA'][(players.Age == x) & ((players.Pos == 'PS') | (players.Pos.values == 'SG'))]))

age_df['3P ratio g'] = age_df['3PA_guard']/(age_df['2PA_guard'] + age_df['3PA_guard'])

age_df['2P ratio g'] = 1 - age_df['3P ratio g']



plt.bar(range(len(age_df['3P ratio g'])),age_df['2P ratio g'])

plt.bar(range(len(age_df['3P ratio g'])),age_df['3P ratio g'],bottom = age_df['2P ratio g'])

plt.xticks(range(18),np.arange(20,38,1))

plt.xlabel('Age')

plt.ylabel('Share of scoring attempts')

plt.title('The share of 3 pointers goes up with age')
shots['Player_Age'] = shots.player_name.apply(lambda x: players.Age[players.Name == x].values[0] if len(players.Age[players.Name == x].values)>0 else 0) 

shots['Pos'] = shots.player_name.apply(lambda x: players.Pos[players.Name == x].values[0] if len(players.Pos[players.Name == x].values)>0 else 0) 
sns.distplot(shots.SHOT_DIST[(shots.Player_Age == 21) & ((shots.Pos == 'PS') | (shots.Pos.values == 'SG'))], bins = np.arange(1,30,1))

sns.distplot(shots.SHOT_DIST[(shots.Player_Age == 37) & ((shots.Pos == 'PS') | (shots.Pos.values == 'SG'))], bins = np.arange(1,30,1))



plt.legend(['Age 21','Age 37'])

plt.title('Guards shooting distance distribution')