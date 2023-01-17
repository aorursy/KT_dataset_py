%matplotlib inline
import pandas as pd

import numpy as np

from pylab import *

import seaborn as sns
#load tables 

batting_reg = pd.DataFrame.from_csv('../input/batting.csv', index_col = None, encoding = 'utf-8')

pitching_reg = pd.DataFrame.from_csv('../input/pitching.csv', index_col = None, encoding = 'utf-8')

player = pd.DataFrame.from_csv('../input/player.csv', index_col = None, encoding = 'utf-8')
#Cleaning up PLayer info

player_name = player[['player_id', 'name_first', 'name_last', 'birth_year']]

player_name = player_name[pd.notnull(player_name['birth_year'])]

player_name['birth_year'] = player_name['birth_year'].astype('int')
#Hitters' age

batting = batting_reg[['player_id', 'year', 'team_id']]

batting['year'] = batting['year'].astype('int')

batting.set_index('player_id')

batting_age = pd.merge(batting, player_name, on = 'player_id')

batting_age['age'] = batting_age['year'].sub(batting_age['birth_year'])

batting_age.drop_duplicates().head()
#pitching age

pitching = pitching_reg[['player_id', 'year', 'team_id']]

pitching['year'] = pitching['year'].astype('int')

pitching.set_index('player_id')

pitching_age = pd.merge(pitching, player_name, on = 'player_id')

pitching_age['age'] = pitching_age['year'].sub(pitching_age['birth_year'])

pitching_age.drop_duplicates().head()
#Average Batters Age

batting_avg_age = batting_age.groupby('year')['age'].mean().reset_index()

b_year_age = batting_avg_age['year']

b_age = batting_avg_age['age']



#Average Pitchers Age

pitching_avg_age = pitching_age.groupby('year')['age'].mean().reset_index()

p_year_age = pitching_avg_age['year']

p_age = pitching_avg_age['age']
#function to convert year to decade

def dec(x):

    return int(x / 10) * 10



#batting age by decade

b_dec_age = batting_age[['year', 'age']]

b_dec_age['year'] = b_dec_age['year'].map(dec)

#pitching age

p_dec_age = pitching_age[['year', 'age']]

p_dec_age['year'] = p_dec_age['year'].map(dec)
fig = plt.figure(figsize=(8,4), dpi=100)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (15,7), sharex = True)



ax1.scatter(b_year_age, b_age, color = 'red')

ax1.set_ylabel('Age')

ax1.set_title('Average Age of MLB Hitters')



ax2.scatter(p_year_age, p_age)

ax2.set_xlabel('Year')

ax2.set_ylabel('Age')

ax2.set_title('Average Age of MLB Pitchers')
bins = np.arange(18, 40, 2)

decade = sns.FacetGrid(b_dec_age, col = 'year', col_wrap = 5, sharey = False)

decade = decade.map(plt.hist, 'age', bins = bins, color = 'red')
bins = np.arange(18, 40, 2)

decade_p = sns.FacetGrid(p_dec_age, col = 'year', col_wrap = 5, sharey = False)

decade_p = decade_p.map(plt.hist, 'age', bins = bins)
teams = pd.DataFrame.from_csv('../input/team.csv', index_col = None, encoding = 'utf-8')

teams = teams.rename(columns = {'year':'team_year'})

teams.head()
t = teams[['name', 'franchise_id', 'team_id', 'team_year', 'w', 'l']]

bat_age = batting_age

pit_age = pitching_age



bat_age = bat_age.merge(t, on = 'team_id')

pit_age = pit_age.merge(t, on = 'team_id')



b_age = bat_age.loc[bat_age['year'] >= 1961, :].reset_index()

p_age = pit_age.loc[pit_age['year'] >= 1961, :].reset_index()



b_age = b_age.groupby(['franchise_id', 'year'])['age', 'w'].mean().reset_index()

p_age = p_age.groupby(['franchise_id', 'year'])['age', 'w'].mean().reset_index()
fig = plt.figure(figsize=(8,4), dpi=100)
fig, (ax1, ax2) = plt.subplots(1,2, figsize = (17, 9), sharex = True)



sns.regplot(x="age", y="w", data=b_age, ax=ax1, color='red')

ax1.set_xlabel('Average Age of Team')

ax1.set_ylabel('Average Wins per Season')

ax1.set_title('HITTING Average Wins vs. Average Age')



sns.regplot(x='age', y='w', data=p_age, ax=ax2)

ax2.set_xlabel('Average Age of Team')

ax2.set_ylabel('Average Wins per Season')

ax2.set_title('PITCHING Average Wins vs. Average Age')

None
fig, ax1 = plt.subplots(figsize = (15, 8), sharex = True)

bat_age = bat_age.loc[bat_age['year'] == 2015, :]

pit_age = pit_age.loc[pit_age['year'] == 2015, :]

#Boxplot

sns.boxplot(x='franchise_id', y='age', data=bat_age, ax=ax1)

ax1.set_xlabel('Team')

ax1.set_ylabel('Player Age')

ax1.set_title('Player Ages in 2015')

None
t_2015 = t.loc[t['team_year'] == 2015, :]

#t_2015

fran_id = list(t_2015['franchise_id'])

xlabel = list(range(30))

fig, ax = plt.subplots(figsize = (15, 8))

plt.bar(xlabel, t_2015['w'])

plt.xticks(xlabel, fran_id)

ax.set_xlabel('Teams')

ax.set_ylabel('Wins')

ax.set_title('Team Wins in 2015')

None