import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print(os.listdir("../input"))
team_df = pd.read_csv('../input/team_info.csv')
team_df.head()
game_df.head()
"""
Add home and away team names.
"""
game_df = pd.read_csv('../input/game.csv')
game_df = game_df.merge(team_df[['team_id', 'teamName']],
              left_on='home_team_id', right_on='team_id') \
    .merge(team_df[['team_id', 'teamName']], left_on='away_team_id',
           right_on='team_id', suffixes=('home','away'))
game_df.head()
game_df[['away_goals','home_goals']].plot(kind='hist', figsize=(15,5), bins=10, alpha=0.5, title='Distribution of Home vs. Away Goals')
game_df.groupby('teamNamehome').mean()['home_goals'] \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 8), title='Average Goals Scored in Home Games')
plt.show()
game_df.groupby('teamNameaway').mean()['away_goals'] \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 8), title='Average Goals Scored in Away Games')
plt.show()
game_df.groupby('teamNamehome').mean()['away_goals'] \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 8), title='Average Goals Allowed in Home Games')
plt.show()
game_df.groupby('teamNameaway').mean()['home_goals'] \
    .sort_values() \
    .plot(kind='barh', figsize=(15, 8), title='Average Goals Allowed in Away Games')
plt.show()
game_df['point_diff'] = game_df['home_goals'] - game_df['away_goals']
game_df['point_diff'].plot(kind='hist',
                           bins=18,
                           title='NHL Point Differential (Negative Home team Loses, Positive Home team Wins)',
                           xlim=(-10,10))
#Biggest Blowout was by 10 points
game_df['point_diff'].abs().max()
# Blowout game:
game_df.loc[game_df['point_diff'] == 10]
game_df['point_diff_type'] = game_df['point_diff'].abs().apply(lambda x: 'Blowout' if x>=3 else ('Normal' if x>=2 else 'Tight'))
# Create one dataframe with the point 
point_diff_team = pd.concat([game_df[['teamNamehome','point_diff_type','point_diff','date_time']].rename(columns={'teamNamehome':'team'}),
    game_df[['teamNameaway','point_diff_type','point_diff','date_time']].rename(columns={'teamNameaway':'team'})])
point_diff_team['date_time'] = pd.to_datetime(point_diff_team['date_time'])
for team, data in point_diff_team.groupby('team'):
    data.groupby(data['date_time'].dt.year).mean()['point_diff'].plot(kind='line', title='{} Average Point Diff By Year'.format(team), figsize=(15,2))
    plt.show()
