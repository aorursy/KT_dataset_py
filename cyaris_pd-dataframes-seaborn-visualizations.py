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
reg_win_percentage = reg_win_percentage.rename(columns = {0 : 'win_percentage', 1: 'home_win_percentage', 2: 'away_win_percentage'})
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
import seaborn as sns
# only game variables (errors, hits, runs, wins, losses, and hours) have correlations above 0.25
df.corr()[(df.corr() > .25) & (df.corr() != 1)]
# only game variables (errors, hits, runs, wins, losses, and hours) have correlations above 0.25
plt.figure(figsize = (15, 8))

sns.heatmap(df.corr(),cmap='coolwarm', annot=True)
plt.title('df.corr()');
pairplot_numbers = df.copy()
pairplot_numbers.drop(['away_team'],axis=1,inplace=True)
pairplot_numbers.drop(['field_type'],axis=1,inplace=True)
pairplot_numbers.drop(['game_type'],axis=1,inplace=True)
pairplot_numbers.drop(['home_team'],axis=1,inplace=True)
pairplot_numbers.drop(['start_time'],axis=1,inplace=True)
pairplot_numbers.drop(['day_of_week'],axis=1,inplace=True)
pairplot_numbers.drop(['wind_direction'],axis=1,inplace=True)
pairplot_numbers.drop(['sky'],axis=1,inplace=True)
pairplot_numbers.drop(['season'],axis=1,inplace=True)
pairplot_numbers.drop(['home_team_outcome'],axis=1,inplace=True)
pairplot_numbers.drop(['venue'],axis=1,inplace=True)
pairplot_numbers.drop(['date'],axis=1,inplace=True)
pairplot_numbers.drop(['away_team_errors'],axis=1,inplace=True)
pairplot_numbers.drop(['home_team_errors'],axis=1,inplace=True)
pairplot_numbers.drop(['away_team_hits'],axis=1,inplace=True)
pairplot_numbers.drop(['home_team_hits'],axis=1,inplace=True)
pairplot_numbers.drop(['home_team_win'],axis=1,inplace=True)
pairplot_numbers.drop(['home_team_loss'],axis=1,inplace=True)
pairplot_numbers.drop(['total_runs'],axis=1,inplace=True)
pairplot_numbers['attendance'] = pairplot_numbers['attendance'].fillna(0)
ax = sns.pairplot(pairplot_numbers)
plt.figure(figsize = (12, 6))

ax = sns.violinplot(x=by_sky['sky'], y=by_sky['home_win_percentage'], data=by_sky)
ax.set_xlabel('Sky Condition')
ax.set_ylabel('Home Team Win Percentage')

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.boxplot(x=df['sky'], y=df['total_runs'], data=df)
ax.set_xlabel('Sky Condition')
ax.set_ylabel('Total Runs Scored')

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.violinplot(x=by_sky['sky'], y=by_sky['home_win_percentage'], hue='game_type', data=df, split=True)
ax.set_xlabel('Sky Conition')
ax.set_ylabel('Home Team Win Percentage')

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.countplot(x=df['sky'], hue=df['home_team_win'], data=df)
ax.set_xlabel('Sky Condition')
ax.set_ylabel('Games')
ax.legend(['Home Team Loss', 'Home Team Win'])

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.violinplot(x=by_wind_direction['wind_direction'], y=by_wind_direction['home_win_percentage'], data=by_wind_direction)
ax.set_xlabel('Wind Direction')
ax.set_ylabel('Home Team Win Percentage')
ax.set_xticklabels(by_wind_direction['wind_direction'].unique(), rotation = 45, ha="right")

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.violinplot(x=by_wind_direction['wind_direction'], y=by_wind_direction['home_win_percentage'], hue='game_type', data=df, split=True)
ax.set_xlabel('Wind Direction')
ax.set_ylabel('Home Team Win Percentage')
ax.set_xticklabels(by_wind_direction['wind_direction'].unique(), rotation = 45, ha="right")

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.boxplot(x=df['wind_direction'], y=df['total_runs'], data=df)
ax.set_xlabel('Wind Direction')
ax.set_ylabel('Total Runs Scored')
ax.set_xticklabels(df['wind_direction'].unique(), rotation = 45, ha="right")

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.boxplot(x=df['wind_direction'], y=df['total_runs'], hue='game_type', data=df)
ax.set_xlabel('Wind Direction')
ax.set_ylabel('Total Runs Scored')
ax.set_xticklabels(df['wind_direction'].unique(), rotation = 45, ha="right")

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.violinplot(x=df['wind_direction'], y=df['total_runs'], data=df)
ax.set_xlabel('Wind Direction')
ax.set_ylabel('Total Runs Scored')
ax.set_xticklabels(df['wind_direction'].unique(), rotation = 45, ha="right")

plt.tight_layout()
plt.figure(figsize = (15, 8))

ax = sns.countplot(x=df['wind_direction'], hue=df['home_team_win'], data=df)
ax.set_xlabel('Wind Direction')
ax.set_ylabel('Games')
ax.set_xticklabels(df['wind_direction'].unique(), rotation = 45, ha="right")
ax.legend(['Home Team Loss', 'Home Team Win'])

plt.tight_layout()
ax = sns.jointplot(x='wind_speed', y='home_win_percentage', data=by_venue, kind='scatter')
ax.set_axis_labels('Wind Speed', 'Home Team Win Percentage')
ax.plot(sns.regplot, sns.distplot);
plt.figure(figsize = (12, 6))

ax = sns.countplot(x=df['wind_speed'], hue=df['home_team_win'], data=df)
ax.set_xlabel('Wind Speed')
ax.set_ylabel('Games')
ax.legend(['Home Team Loss', 'Home Team Win'])

plt.tight_layout()
plt.figure(figsize = (12, 6))

ax = sns.boxplot(x=df['wind_speed'], y=df['total_runs'], data=df)
ax.set_xlabel('Wind Speed')
ax.set_ylabel('Total Runs Scored')

plt.tight_layout()
ax = sns.jointplot(x='attendance', y='home_win_percentage', data=by_venue, kind='scatter')
ax.set_axis_labels('Attendance', 'Home Team Win Percentage')
ax.plot(sns.regplot, sns.distplot);
ax = sns.jointplot(x=df[df['home_team_win']==1]['attendance'], y=df[df['home_team_win']==1]['home_team_runs'], data=df, kind = 'scatter')
ax.set_axis_labels('Attendance', 'Home Team Runs')
ax.plot(sns.regplot, sns.distplot)
ax.plot_joint(plt.scatter, edgecolor="white");
ax = sns.jointplot(x=df[df['home_team_win']==0]['attendance'], y=df[df['home_team_win']==0]['home_team_runs'], data=df, kind='scatter')
ax.set_axis_labels('Attendance', 'Home Team Runs')
ax.plot(sns.regplot, sns.distplot)
ax.plot_joint(plt.scatter, edgecolor="white");
ax = sns.JointGrid(x=df[df['home_team_win']==1]['attendance'], y=df[df['home_team_win']==1]['home_team_runs'], data=df, space=0)
ax.set_axis_labels('Attendance', 'Home Team Runs')
ax.plot_joint(sns.kdeplot, cmap="Blues_d")
ax.plot_marginals(sns.kdeplot, shade=True);
ax = sns.JointGrid(x=df[df['home_team_win']==0]['attendance'], y=df[df['home_team_win']==0]['home_team_runs'], data=df, space=0)
ax.set_axis_labels('Attendance', 'Home Team Runs')
ax.plot_joint(sns.kdeplot, cmap="Blues_d")
ax.plot_marginals(sns.kdeplot, shade=True);
ax = sns.lmplot(x='attendance', y="home_team_runs", col="home_team_outcome",data=df)
ax.set_axis_labels('Attendance', 'Home Team Runs');
plt.figure(figsize = (6, 4))

ax = sns.countplot(x=df['game_type'], hue=df['home_team_win'], data=df)
ax.set_xlabel('Game Type')
ax.set_ylabel('Games')
ax.legend(['Home Team Loss', 'Home Team Win'])

plt.tight_layout()
plt.figure(figsize = (6, 4))

ax = sns.countplot(x=df['field_type'], hue=df['home_team_win'], data=df)
ax.set_xlabel('Field Type')
ax.set_ylabel('Games')
ax.legend(['Home Team Loss', 'Home Team Win'])

plt.tight_layout()
ax = sns.jointplot(x='temperature', y='home_win_percentage', data=by_venue, kind='scatter')
ax.set_axis_labels('Temperature', 'Home Team Win Percentage')
ax.plot(sns.regplot, sns.distplot);
ax = sns.jointplot(x='game_hours_dec', y='home_win_percentage', data=by_venue, kind='scatter')
ax.set_axis_labels("Game Length (Hours)", 'Home Team Win Percentage')
ax.plot(sns.regplot, sns.distplot);