%matplotlib inline



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)

sns.set()



print('All dependencies installed')
def lower_case(dataset):

    dataset.columns = [col.lower() for col in dataset.columns]

    return dataset



game_data = pd.read_csv('../input/game_data.csv', parse_dates=True)

game_data = lower_case(game_data)

print(game_data.shape)

game_data.head()
def num_missing(x):

  return print('Missing values per column:\n', np.sum(x.isnull()))



num_missing(game_data)
stadium_type = game_data['stadiumtype'].value_counts()

turf = game_data['turf'].value_counts()

game_weather = game_data['gameweather'].value_counts()

temperature = game_data['temperature'].value_counts()

outdoor_weather = game_data['outdoorweather'].value_counts



print(stadium_type, '\n', '-'*50, '\n', turf, '\n', '-'*50, '\n', game_weather, '\n', '-'*50, '\n', temperature,  '\n', '-'*50, '\n', outdoor_weather)
game_data = game_data.drop(columns=['outdoorweather', 'gameweather'], axis=1)

game_data.info()
game_data.info()
category_columns = ['season_type', 'stadiumtype', 'turf']

float_columns = ['temperature']



game_data[category_columns] = game_data[category_columns].astype('category')

game_data[float_columns] = game_data[float_columns].astype(float)

date = pd.to_datetime(game_data['game_date'].str.split(expand=True)[0], format='%Y-%m-%d')



game_data.info()
plt.figure(figsize=(20, 9))



_ = sns.scatterplot(x='home_team', y='visit_team', hue='week',data=game_data)

plt.xticks(rotation=90, fontsize=14)

plt.yticks(fontsize=15)

plt.xlabel('Visiting Team')

plt.ylabel('Home Team')



plt.show()
video_footage_injury = pd.read_csv('../input/video_footage-injury.csv', parse_dates=True)

video_footage_injury = lower_case(video_footage_injury)

print(game_data.shape)

video_footage_injury.head()
num_missing(video_footage_injury)
video_footage_injury.info()
plt.figure(figsize=(20, 7.5))



plt.subplot(1, 2, 1)

_ = sns.countplot(video_footage_injury['type'])

plt.title('Concussions per season type:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Type of season', fontsize=15)

plt.ylabel('Total amount of concussions', fontsize=15)



plt.subplot(1, 2, 2)

_ = sns.countplot(video_footage_injury['season'])

plt.title('Concussions per year:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Year', fontsize=15)

plt.ylabel('Total amount of concussions', fontsize=15)



plt.tight_layout()

plt.show()
plt.figure(figsize=(20, 10))



plt.subplot(2, 1, 1)

_ = sns.swarmplot(x='week', y='type', hue='qtr', data=video_footage_injury)

plt.title('Concussions per week:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Week', fontsize=15)

plt.ylabel('Season Type', fontsize=15)



plt.subplot(2, 1, 2)

_ = sns.boxplot(x='qtr', y='week', data=video_footage_injury)

plt.title('Concussions per quarter:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Week', fontsize=15)

plt.ylabel('')



plt.tight_layout()

plt.show()
video_review = pd.read_csv('../input/video_review.csv')

video_review = lower_case(video_review)

print(video_review.shape)

video_review.head()
num_missing(video_review)
video_review = video_review.drop(columns=['primary_partner_gsisid'], axis=1)

video_review = video_review.dropna()

video_review.info()
video_review.info()
category_columns = ['player_activity_derived', 'turnover_related', 'primary_impact_type', 'primary_partner_activity_derived', 'friendly_fire']



video_review[category_columns] = video_review[category_columns].astype('category')

video_review.info()
plt.figure(figsize=(20, 7.5))



plt.subplot(1, 2, 1)

_ = sns.countplot(video_review['player_activity_derived'])

plt.title('Concussions related incidents:', fontsize=20)

plt.xlabel('')

plt.ylabel('')

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)



plt.subplot(1, 2, 2)

_ = sns.countplot(video_review['friendly_fire'])

plt.title('Incidents from same team or opposing:', fontsize=20)

plt.xlabel('')

plt.ylabel('')

plt.xticks([0, 1, 2], ['Opposing Team', 'Unclear', 'Same Team'], fontsize=15)

plt.yticks(fontsize=15)



plt.tight_layout()

plt.show()
video_footage_control = pd.read_csv('../input/video_footage-control.csv')

video_footage_control = lower_case(video_footage_control)

print(video_footage_control.shape)

video_footage_control.head()
num_missing(video_footage_control)
video_footage_control.info()
quarter_count = video_footage_control['qtr'].value_counts()



plt.figure(figsize=(20, 7.5))



_ = sns.scatterplot(x='qtr', y='home_team', data=video_footage_control)

plt.title('Distributions of injuries per quarter:', fontsize=20)

plt.xlabel('Quarter', fontsize=15)

plt.ylabel('')

plt.xticks([1, 2, 3, 4], fontsize=15)

plt.yticks(fontsize=15)



plt.show()
play_information = pd.read_csv('../input/play_information.csv')

play_information = lower_case(play_information)

print(play_information.shape)

play_information.head()
num_missing(play_information)
play_information.info()
category_columns = ['season_type']

play_information[category_columns] = play_information[category_columns].astype('category')

play_information['game_clock'] = pd.to_datetime(play_information['game_clock'], format='%H:%M').dt.time

play_information.info()

play_information.head()
plt.figure(figsize=(20, 7.5))



_ = sns.countplot(play_information['week'])

# _ = sns.swarmplot(x='Week', y='YardLine', data=play_information, hue='Poss_Team')

plt.title('Total amount of plays per week:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Week', fontsize=15)

plt.ylabel('')



plt.show()
play_player_role_data = pd.read_csv('../input/play_player_role_data.csv')

play_player_role_data = lower_case(play_player_role_data)

print(play_player_role_data.shape)

play_player_role_data.head()
num_missing(play_player_role_data)
play_player_role_data.info()
plt.figure(figsize=(20, 7.5))



_ = sns.countplot(play_player_role_data['role'])

plt.title('Counting the total amount of players in each position per play:', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Role', fontsize=15)

plt.ylabel('')



plt.show()
player_punt_data = pd.read_csv('../input/player_punt_data.csv')

player_punt_data = lower_case(player_punt_data)

print(player_punt_data.shape)

player_punt_data.head()
num_missing(player_punt_data)
player_punt_data.info()
plt.figure(figsize=(20, 7.5))



_ = sns.countplot(player_punt_data['position'])

plt.title('Count of each individual position:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Positions', fontsize=15)

plt.ylabel('')



plt.show()
player_data = pd.merge(play_player_role_data, player_punt_data)

player_data = pd.merge(player_data, play_information)

player_data = pd.merge(player_data, video_review)

player_data.head()
player_data = player_data.sort_values('poss_team')



plt.figure(figsize=(20, 7.5))



_ = sns.countplot(player_data['poss_team'])

plt.title('Teams that suffered injuries from punts:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('', fontsize=15)

plt.ylabel('Possesing Team', fontsize=15)



plt.show()
plt.figure(figsize=(20, 7.5))



_ = sns.countplot(y=player_data['player_activity_derived'], hue=player_data['poss_team'])

plt.title('Activity when suffering injuries from punts:', fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('', fontsize=15)

plt.ylabel('Player Activity Derived', fontsize=15)

plt.legend(fontsize=15)



plt.show()
plt.figure(figsize=(20, 7.5))



_ = sns.countplot(player_data['home_team_visit_team'], hue=player_data['player_activity_derived'])

plt.title('Analyzing if teams play more aggressively against others:', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Possesing Team', fontsize=15)

plt.ylabel('')

plt.legend(fontsize=15)



plt.show()
player_data['yard'] = player_data['yardline'].str.split(' ').str[1].astype(int)

player_data.sort_values(by=['yard'], inplace=True)
plt.figure(figsize=(20, 10))



plt.subplot(2, 1, 1)

_ = sns.countplot(player_data['yard'])

plt.title('Analyzing location of injuries:', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Yard', fontsize=15)

plt.ylabel('')

plt.legend(fontsize=15, loc=1)



plt.subplot(2, 1, 2)

_ = sns.swarmplot(x='poss_team', y='yard', data=player_data, hue='quarter', s=7.5, palette=['Green', 'Blue', 'Orange', 'Red'])

plt.title('Analyzing location of injuries by team:', fontsize=20)

plt.xticks(rotation=90, fontsize=15)

plt.yticks(fontsize=15)

plt.xlabel('Possesing Team', fontsize=15)

plt.ylabel('Yard', fontsize=15)

plt.legend(fontsize=15)



plt.tight_layout()

plt.show()
player_data.sort_values(['quarter', 'game_clock'])
plt.figure(figsize=(20, 7.5))



_ = sns.barplot(x='game_clock', y=range(16), data=player_data)

plt.xticks(rotation=90)



plt.show()
play_data = pd.merge(play_information, game_data, on='gamekey')

print(play_data['playdescription'][0])

print(play_data['playdescription'][1])

print(play_data['playdescription'][2])

print(play_data['playdescription'][3])

print(play_data['playdescription'][4])