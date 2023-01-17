import numpy as np
import pandas as pd
df = pd.read_csv('../input/baseball_reference_2016_clean.csv',index_col=0)
df['attendance'] = df['attendance'].astype(float)
df['date'] = pd.to_datetime(df['date'])
df['temperature'] = df['temperature'].astype(float)
df['wind_speed'] = df['wind_speed'].astype(float)
reg_season = df[df['season']=='regular season']
reg_wins = pd.DataFrame(reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['home_team'].value_counts() + reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['away_team'].value_counts())
reg_wins.set_axis(['wins'],axis='columns',inplace=True)
reg_wins.index.name = 'team'
reg_home_wins = pd.DataFrame(reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['home_team'].value_counts())
reg_home_wins.set_axis(['home_wins'],axis='columns',inplace=True)
reg_home_wins.index.name = 'team'
reg_losses = pd.DataFrame(reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['home_team'].value_counts() + reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['away_team'].value_counts())
reg_losses.set_axis(['losses'],axis='columns',inplace=True)
reg_losses.index.name = 'team'
reg_home_losses = pd.DataFrame(reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['home_team'].value_counts())
reg_home_losses.set_axis(['home_losses'],axis='columns',inplace=True)
reg_home_losses.index.name = 'team'
win_percentage = reg_wins.wins/(reg_wins.wins + reg_losses.losses)
home_win_percentage = reg_home_wins.home_wins/(reg_home_wins.home_wins + reg_home_losses.home_losses)
away_win_percentage = (reg_wins.wins - reg_home_wins.home_wins)/((reg_wins.wins - reg_home_wins.home_wins) + (reg_losses.losses - reg_home_losses.home_losses))
outcomes = [reg_wins, reg_home_wins, reg_losses, reg_home_losses, win_percentage, home_win_percentage, away_win_percentage]

reg_win_percentage = pd.concat(outcomes,axis = 1, join = 'outer')
reg_win_percentage.index.name = 'team'
reg_win_percentage = reg_win_percentage.rename(columns={0 : 'win_percentage', 1: 'home_win_percentage', 2: 'away_win_percentage'})
reg_win_percentage.drop(['wins'],axis=1,inplace=True)
reg_win_percentage.drop(['home_wins'],axis=1,inplace=True)
reg_win_percentage.drop(['losses'],axis=1,inplace=True)
reg_win_percentage.drop(['home_losses'],axis=1,inplace=True)
reg_win_percentage = reg_win_percentage.round(2)
reg_win_percentage.head()
aggregations = {
    'venue' : 'count',
    'home_team_win' : 'sum',
    'home_team_loss' : 'sum',
    'attendance' : 'mean',
    'temperature' : 'mean',
    'wind_speed' : 'mean',
    'game_hours_dec' : 'mean'
    }
by_game_type = df[df['season']=='regular season'].groupby(['home_team', 'venue', 'game_type']).agg(aggregations)
by_game_type = by_game_type.rename(columns={'venue' : 'games_played'})
by_game_type['home_win_percentage'] = by_game_type['home_team_win']/(by_game_type['home_team_win'] + by_game_type['home_team_loss'])
by_game_type.drop(['home_team_win'],axis=1,inplace=True)
by_game_type.drop(['home_team_loss'],axis=1,inplace=True)
# removing any venue that did not have at least 80 games played.
# only instance is single Braves game played at Fort Bragg Park.
by_game_type = by_game_type[0:2].append(by_game_type[3:])
by_game_type = by_game_type.round(2)
by_game_type = by_game_type.reset_index()
by_game_type.head()
by_sky = df[df['season']=='regular season'].groupby(['home_team', 'venue', 'sky']).agg(aggregations)
by_sky = by_sky.rename(columns={'venue' : 'games_played'})
by_sky['home_win_percentage'] = by_sky['home_team_win']/(by_sky['home_team_win'] + by_sky['home_team_loss'])
by_sky.drop(['home_team_win'],axis=1,inplace=True)
by_sky.drop(['home_team_loss'],axis=1,inplace=True)
# removing any venue that did not have at least 80 games played.
# only instance is single Braves game played at Fort Bragg Park.
by_sky = by_sky[0:4].append(by_sky[5:])
by_sky = by_sky.round(2)
by_sky = by_sky.reset_index()
by_sky.head()
by_wind_direction = df[df['season']=='regular season'].groupby(['home_team', 'venue', 'wind_direction']).agg(aggregations)
by_wind_direction = by_wind_direction.rename(columns={'venue' : 'games_played'})
by_wind_direction['home_win_percentage'] = by_wind_direction['home_team_win']/(by_wind_direction['home_team_win'] + by_wind_direction['home_team_loss'])
by_wind_direction.drop(['home_team_win'],axis=1,inplace=True)
by_wind_direction.drop(['home_team_loss'],axis=1,inplace=True)
# removing any venue that did not have at least 80 games played.
# only instance is single Braves game played at Fort Bragg Park.
by_wind_direction = by_wind_direction[0:7].append(by_wind_direction[8:])
by_wind_direction = by_wind_direction.round(2)
by_wind_direction = by_wind_direction.reset_index()
by_wind_direction.head()
by_venue = df[df['season']=='regular season'].groupby(['home_team', 'venue']).agg(aggregations)
by_venue = by_venue.rename(columns={'venue' : 'games_played'})
by_venue['home_win_percentage'] = by_venue['home_team_win']/(by_venue['home_team_win'] + by_venue['home_team_loss'])
by_venue.drop(['home_team_win'],axis=1,inplace=True)
by_venue.drop(['home_team_loss'],axis=1,inplace=True)
# removing any venue that did not have at least 80 games played.
# only instance is single Braves game played at Fort Bragg Park.
by_venue = by_venue[0:1].append(by_venue[2:])
by_venue = by_venue.round(2)
by_venue = by_venue.reset_index()
by_venue['home_team'] = by_venue['home_team'].astype(str)
by_venue['venue'] = by_venue['venue'].astype(str)
by_venue.head()
import matplotlib.pyplot as plt
%matplotlib inline
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 25))

axes[0].bar(by_venue['venue'], by_venue['attendance'], color=['red', 'blue'], alpha=0.5)
axes[0].set_ylabel('Attendamce')
axes[0].set_title('Average Attendance by Venue')
axes[0].set_xticklabels(by_venue['venue'], rotation=45, ha="right")

axes[1].bar(by_venue['venue'], by_venue['game_hours_dec'], color=['red', 'blue'], alpha=0.5)
axes[1].set_ylabel("Game Length (Hours)")
axes[1].set_title("Average Game Length by Venue")
axes[1].set_xticklabels(by_venue['venue'], rotation=45, ha="right")

axes[2].bar(by_venue['venue'], by_venue['temperature'], color=['red', 'blue'], alpha=0.5)
axes[2].set_ylabel('Temperature')
axes[2].set_title("Average Temperature by Venue")
axes[2].set_xticklabels(by_venue['venue'], rotation=45, ha="right")

axes[3].bar(by_venue['venue'], by_venue['wind_speed'], color=['red', 'blue'], alpha=0.5)
axes[3].set_ylabel('Wind Speed')
axes[3].set_title("Average Wind Speed by Venue")
axes[3].set_xticklabels(by_venue['venue'], rotation=45, ha="right")

fig.tight_layout()
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 20))

axes[0].scatter(by_venue['temperature'],by_venue['home_win_percentage'],c=by_venue['home_win_percentage'])
axes[0].set_xlabel('Temperature')
axes[0].set_ylabel('Home Win Percentage')
axes[0].set_title('Correlation Between Temperature and Home Win Percentage')

axes[1].scatter(by_venue['wind_speed'],by_venue['home_win_percentage'],c=by_venue['home_win_percentage'])
axes[1].set_xlabel('Wind Speed')
axes[1].set_ylabel('Home Win Percentage')
axes[1].set_title('Correlation Between Wind Speed and Home Win Percentage')

axes[2].scatter(by_venue['attendance'],by_venue['home_win_percentage'],c=by_venue['home_win_percentage'])
axes[2].set_xlabel('Attendance')
axes[2].set_ylabel('Home Win Percentage')
axes[2].set_title('Correlation Between Attendance and Home Win Percentage')

axes[3].scatter(by_venue['game_hours_dec'],by_venue['home_win_percentage'],c=by_venue['home_win_percentage'])
axes[3].set_xlabel("Game Length (Hours)")
axes[3].set_ylabel('Home Win Percentage')
axes[3].set_title("Correlation Between Game Length and Home Win Percentage")

fig.tight_layout()
fig, axes = plt.subplots(nrows=14, ncols=1, figsize=(12, 80))

axes[0].scatter(df[df['home_team_win']==1]['game_hours_dec'],df[df['home_team_win']==1]['attendance'],c='aqua',alpha=0.5)
axes[0].scatter(df[df['home_team_win']==0]['game_hours_dec'],df[df['home_team_win']==0]['attendance'],c='coral',alpha=0.5)
axes[0].set_xlabel("Game Length (Hours)")
axes[0].set_ylabel('Attendance')
axes[0].set_title('Correlation Between Game Length and Attendance')
axes[0].legend(['Home Team Win','Home Team Loss'])

axes[1].scatter(df[df['home_team_win']==1]['attendance'],df[df['home_team_win']==1]['total_runs'],c='aqua',alpha=0.5)
axes[1].scatter(df[df['home_team_win']==0]['attendance'],df[df['home_team_win']==0]['total_runs'],c='coral',alpha=0.5)
axes[1].set_xlabel('Attendance')
axes[1].set_ylabel('Total Runs')
axes[1].set_title('Correlation Between Attendance and Total Runs')
axes[1].legend(['Home Team Win','Home Team Loss'])

axes[2].scatter(df[df['home_team_win']==1]['game_hours_dec'],df[df['home_team_win']==1]['total_runs'],c='aqua',alpha=0.5)
axes[2].scatter(df[df['home_team_win']==0]['game_hours_dec'],df[df['home_team_win']==0]['total_runs'],c='coral',alpha=0.5)
axes[2].set_xlabel("Game Length (Hours)")
axes[2].set_ylabel('Total Runs')
axes[2].set_title('Correlation Between Game Length and Total Runs')
axes[2].legend(['Home Team Win','Home Team Loss'])

axes[3].scatter(df[df['home_team_win']==1]['game_hours_dec'],df[df['home_team_win']==1]['temperature'],c='aqua',alpha=0.5)
axes[3].scatter(df[df['home_team_win']==0]['game_hours_dec'],df[df['home_team_win']==0]['temperature'],c='coral',alpha=0.5)
axes[3].set_xlabel("Game Length (Hours")
axes[3].set_ylabel('Temperature')
axes[3].set_title('Correlation Between Game Length Temperature')
axes[3].legend(['Home Team Win', 'Home Team Loss'])

axes[4].scatter(df[df['home_team_win']==1]['temperature'],df[df['home_team_win']==1]['total_runs'],c='aqua',alpha=0.5)
axes[4].scatter(df[df['home_team_win']==0]['temperature'],df[df['home_team_win']==0]['total_runs'],c='coral',alpha=0.5)
axes[4].set_xlabel('Temperature')
axes[4].set_ylabel('Total Runs')
axes[4].set_title('Correlation Between Temperature and Total Runs')
axes[4].legend(['Home Team Win', 'Home Team Loss'])

axes[5].scatter(df[df['home_team_win']==1]['game_hours_dec'],df[df['home_team_win']==1]['wind_speed'],c='aqua',alpha=0.5)
axes[5].scatter(df[df['home_team_win']==0]['game_hours_dec'],df[df['home_team_win']==0]['wind_speed'],c='coral',alpha=0.5)
axes[5].set_xlabel("Game Length (Hours")
axes[5].set_ylabel('Wind Speed')
axes[5].set_title('Correlation Between Game Length and Wind Speed')
axes[5].legend(['Home Team Win','Home Team Loss'])

axes[6].scatter(df[df['home_team_win']==1]['wind_speed'],df[df['home_team_win']==1]['total_runs'],c='aqua',alpha=0.5)
axes[6].scatter(df[df['home_team_win']==0]['wind_speed'],df[df['home_team_win']==0]['total_runs'],c='coral',alpha=0.5)
axes[6].set_xlabel('Wind Speed')
axes[6].set_ylabel('Total Runs')
axes[6].set_title('Correlation Between Wind Speed and Total Runs')
axes[6].legend(['Home Team Win','Home Team Loss'])

axes[7].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['attendance'],c='aqua',alpha=0.5)
axes[7].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['attendance'],c='coral',alpha=0.5)
axes[7].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['attendance'],c='orchid',alpha=0.5)
axes[7].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['attendance'],c='grey',alpha=0.5)
axes[7].set_xlabel("Game Length (Hours)")
axes[7].set_ylabel('Attendance')
axes[7].set_title('Correlation Between Game Length and Attendance')
axes[7].legend(["Home Team Win (Day Game)","Home Team Win (Night Game)","Home Team Loss (Day Game)","Home Team Loss (Night Game)"])

axes[8].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['attendance'],df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['total_runs'],c='aqua',alpha=0.5)
axes[8].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['attendance'],df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['total_runs'],c='coral',alpha=0.5)
axes[8].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['attendance'],df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['total_runs'],c='orchid',alpha=0.5)
axes[8].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['attendance'],df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['total_runs'],c='grey',alpha=0.5)
axes[8].set_xlabel('Attendance')
axes[8].set_ylabel('Total Runs')
axes[8].set_title('Correlation Between Attendance and Total Runs')
axes[8].legend(["Home Team Win (Day Game)","Home Team Win (Night Game)","Home Team Loss (Day Game)","Home Team Loss (Night Game)"])

axes[9].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['total_runs'],c='aqua')
axes[9].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['total_runs'],c='coral')
axes[9].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['total_runs'],c='orchid')
axes[9].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['total_runs'],c='grey',alpha=0.5)
axes[9].set_xlabel("Game Length (Hours)")
axes[9].set_ylabel('Total Runs')
axes[9].set_title('Correlation Between Game Length and Total Runs')
axes[9].legend(["Home Team Win (Day Game)","Home Team Win (Night Game)","Home Team Loss (Day Game)","Home Team Loss (Night Game)"])

axes[10].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['temperature'],c='aqua')
axes[10].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['temperature'],c='coral')
axes[10].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['temperature'],c='orchid')
axes[10].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['temperature'],c='grey',alpha=0.5)
axes[10].set_xlabel("Game Length (Hours)")
axes[10].set_ylabel('Temperature')
axes[10].set_title('Correlation Between Game Length and Temperature')
axes[10].legend(["Home Team Win (Day Game)","Home Team Win (Night Game)","Home Team Loss (Day Game)","Home Team Loss (Night Game)"])

axes[11].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['temperature'],df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['total_runs'],c='aqua')
axes[11].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['temperature'],df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['total_runs'],c='coral')
axes[11].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['temperature'],df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['total_runs'],c='orchid')
axes[11].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['temperature'],df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['total_runs'],c='grey',alpha=0.5)
axes[11].set_xlabel('Temperature')
axes[11].set_ylabel('Total Runs')
axes[11].set_title('Correlation Between Temperature and Total Runs')
axes[11].legend(["Home Team Win (Day Game)","Home Team Win (Night Game)","Home Team Loss (Day Game)","Home Team Loss (Night Game)"])

axes[12].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['wind_speed'],c='aqua')
axes[12].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['wind_speed'],c='coral')
axes[12].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['wind_speed'],c='orchid')
axes[12].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['game_hours_dec'],df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['wind_speed'],c='grey',alpha=0.5)
axes[12].set_xlabel('Game Length (Hours)')
axes[12].set_ylabel("Wind Speed")
axes[12].set_title("Correlation Between Game Length and Wind Speed")
axes[12].legend(["Home Team Win (Day Game)","Home Team Win (Night Game)","Home Team Loss (Day Game)","Home Team Loss (Night Game)"])

axes[13].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['wind_speed'],df[(df['home_team_win']==1)&(df['game_type']=='Day Game')]['total_runs'],c='aqua')
axes[13].scatter(df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['wind_speed'],df[(df['home_team_win']==1)&(df['game_type']=='Night Game')]['total_runs'],c='coral')
axes[13].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['wind_speed'],df[(df['home_team_win']==0)&(df['game_type']=='Day Game')]['total_runs'],c='orchid')
axes[13].scatter(df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['wind_speed'],df[(df['home_team_win']==0)&(df['game_type']=='Night Game')]['total_runs'],c='grey',alpha=0.5)
axes[13].set_xlabel('Wind Speed')
axes[13].set_ylabel('Total Runs')
axes[13].set_title('Correlation Between Wind Speed and Total Runs')
axes[13].legend(["Home Team Win (Day Game)","Home Team Win (Night Game)","Home Team Loss (Day Game)","Home Team Loss (Night Game)"])

fig.tight_layout()