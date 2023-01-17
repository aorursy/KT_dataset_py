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

reg_win_percentage = pd.concat(outcomes,axis=1, join='outer')
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
import plotly.plotly as py
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
init_notebook_mode(connected=True)
cf.go_offline()
aggregations = {
    'home_team_runs' : 'mean',
    'away_team_runs' : 'mean',
    'home_team_win' : 'sum',
    'home_team_loss' : 'sum',
    'attendance' : 'mean',
    'temperature' : 'mean',
    'wind_speed' : 'mean',
    'game_hours_dec' : 'mean'
    }
by_date = df[df['season']=='regular season'].groupby([df['date'].dt.date]).agg(aggregations)
by_date = by_date.reset_index()
by_date.iplot(kind='scatter',x='date',y='attendance',size='C')
by_date = df[df['season']=='regular season'].groupby([df['date'].dt.date]).agg(aggregations)
by_date = by_date.reset_index()
by_date.iplot(kind='scatter',x='date',y='home_team_win',size='C')
by_date = df[df['season']=='regular season'].groupby([df['date'].dt.date]).agg(aggregations)
by_date = by_date.reset_index()
by_date.iplot(kind='scatter',x='date',y=['home_team_runs','away_team_runs'],size='C')
by_date = df[df['season']=='regular season'].groupby([df['date'].dt.date]).agg(aggregations)
by_date = by_date.reset_index()
by_date.iplot(kind='scatter',x='date',y='wind_speed',size='C')
by_date = df[df['season']=='regular season'].groupby([df['date'].dt.date]).agg(aggregations)
by_date = by_date.reset_index()
by_date.iplot(kind='scatter',x='date',y='temperature',size='C')
by_date = df[df['season']=='regular season'].groupby([df['date'].dt.date]).agg(aggregations)
by_date = by_date.reset_index()
by_date.iplot(kind='scatter',x='date',y='game_hours_dec',size='C')