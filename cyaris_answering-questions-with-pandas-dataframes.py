import numpy as np
import pandas as pd
df = pd.read_csv('../input/baseball_reference_2016_clean.csv',index_col=0)
df['attendance'] = df['attendance'].astype(float)
df['date'] = pd.to_datetime(df['date'])
df['temperature'] = df['temperature'].astype(float)
df['wind_speed'] = df['wind_speed'].astype(float)
df[df['attendance']==df['attendance'].max()]
df[df['temperature']==df['temperature'].max()]
df[df['temperature']==df['temperature'].min()]
df[df['game_hours_dec']==df['game_hours_dec'].max()]
df[df['game_hours_dec']==df['game_hours_dec'].min()]
df[df['home_team_runs']==df['away_team_runs']].count()[1]
df[df['date']==df['date'].dt.date.max()]
df[df['attendance']==df['attendance'].max()]
df[df['attendance']==df['attendance'].min()]
df[df['wind_speed']==df['wind_speed'].max()]
df[df['away_team_runs'] + df['home_team_runs']==29]
df['total_errors'] = df['away_team_errors'] + df['home_team_errors']
df[df['total_errors']==df['total_errors'].max()]
df[df['total_runs']==df['total_runs'].max()]
df['away_team'].value_counts() + df['home_team'].value_counts()[1]
reg_season = df[df['season']=='regular season']
reg_wins = pd.DataFrame(reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['home_team'].value_counts() + reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['away_team'].value_counts())
reg_wins.set_axis(['wins'],axis='columns',inplace=True)
reg_wins.index.name = 'team'
reg_wins.sort_values(by='wins',ascending=False).head(1)
reg_home_wins = pd.DataFrame(reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['home_team'].value_counts())
reg_home_wins.set_axis(['home_wins'],axis='columns',inplace=True)
reg_home_wins.index.name = 'team'
reg_home_wins.sort_values(by='home_wins',ascending=False).head(1)
reg_losses = pd.DataFrame(reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['home_team'].value_counts() + reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['away_team'].value_counts())
reg_losses.set_axis(['losses'],axis='columns',inplace=True)
reg_losses.index.name = 'team'
reg_losses.sort_values(by="losses",ascending=False).head(1)
reg_home_losses = pd.DataFrame(reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['home_team'].value_counts())
reg_home_losses.set_axis(['home_losses'],axis='columns',inplace=True)
reg_home_losses.index.name = 'team'
reg_home_losses.sort_values(by='home_losses',ascending=False).head(1)