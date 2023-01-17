# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#A few more imports...

#Regex

import re



#Plotting

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_style('whitegrid')
#Reading the data in and taking a look at what we have

data = pd.read_csv('../input/QBStats_all.csv')

print('\nColumns: ', data.columns, '\n')

print('\nDimensions: ', data.shape, '\n')

print(data.info(), '\n')
#KEY ASSUMPTIONS - These will come in handy later in generating our new rating

#Filtering - Take out games with few attempts and quarterbacks with few games

min_att = 5

min_games = 16



#Scoring - Game level

gm_defeffc = 0.75

gm_factor_weights = {

	'ypa': (25 / 100),

	'effc': (35 / 100),

	'comppct': (20 / 100),

	'ptsacct': (30 / 100),

	'lps': (-10 / 100)}



#Scoring - Season level

season_defeffc = 1

season_factor_weights = {

	'gms_m': 90 / 100,

	'scaled_lg': 20 / 100,

	'scaled_ha': (-10 / 100)}

# Dropping records with less than 10 passes

data = data[data['att'] > min_att]



# Dropping running back records

data = data[data['int'] != '--']
# Create full name attribute for searching and grouping

data['fname'] = data['qb'].str.extract('(^[A-Z]\w{0,})', expand=True)

data['fname'] = data['fname'].str.replace(' ','')

data['lname'] = data['qb'].str.extract('(\s[A-z]{0,})', expand=True)

data['lname'] = data['lname'].str.replace(' ','')

data['lg'] = pd.to_numeric(data['lg'].str.replace('t',''))

data['fullname'] = data['fname'] + ' ' + data['lname']



# Some other cleaning

data = data.drop(['qb','fname','lname'], axis = 1)

data.rename(columns = {'int':'intr'}, inplace = True)

data['td'] = pd.to_numeric(data['td'])

data['intr'] = pd.to_numeric(data['intr'])
# Counting games played and dropping those with less than the minimum threshold of games played

gms_played = data.groupby(['fullname'])['fullname'].agg(['count'])

gms_played.sort_values('count', ascending=False, inplace = True)

drop_qbs = gms_played.index[gms_played['count'] < min_games].tolist()



for item in drop_qbs:

	data = data[data.fullname != item]
# Completion percentage - Incomplete measure of efficiency, but still relevant

data['comppct'] = data['cmp'] / data['att']



# Game points accounted for - Could be a measure of productivity as well as control over the game

data['ptsacct'] = (data['td'] * 6) / data['game_points']



# Loss per sack - Measure of awareness and pocket presence

data['lps'] = 0

data.ix[(data.sack != 0), 'lps'] = data['loss'] / data['sack']



# Efficiency - Tradeoff between touchdowns and interceptions

data['effc'] = 0

data.ix[(data.intr == 0) & (data.td != 0), 'effc'] = data['td']

data.ix[(data.intr == 0) & (data.td == 0), 'effc'] = gm_defeffc

data.ix[(data.intr != 0), 'effc'] = data['td'] / data['intr']
# Check unique values

qbs = data['fullname'].unique()

numpeople = len(qbs)

oldestyear = data['year'].min()

recentyear = data['year'].max()
print('\nGAME STATS\n','---------------')

print('Unique QBs: ', numpeople)

print('Earliest Year: ', oldestyear)

print('Most Recent Year: ', recentyear)

print('Mean YPA: ', data['ypa'].mean())

print('Mean Completion Percentage: ', data['comppct'].mean())

print('Mean Points Accounted For: ', data['ptsacct'].mean())

print('Mean Loss Per Sack: ', data['lps'].mean())

print('Mean Efficiency: ', data['effc'].mean())
# Creating scaled columns for relevant statistics

data['scaled_ypa'] = data['ypa'] / data['ypa'].mean()

data['scaled_effc'] = data['effc'] / data['effc'].mean()

data['scaled_comppct'] = data['comppct'] / data['comppct'].mean()

data['scaled_ptsacct'] = data['ptsacct'] / data['ptsacct'].mean()

data['scaled_lps'] = data['lps'] / data['lps'].mean()

data['gm_score'] = (

	data['scaled_ypa'] * gm_factor_weights['ypa'] + 

	data['scaled_effc'] * gm_factor_weights['effc'] + 

	data['scaled_comppct'] * gm_factor_weights['comppct'] + 

	data['scaled_ptsacct'] * gm_factor_weights['ptsacct'] + 

	data['scaled_lps'] * gm_factor_weights['lps']) * (data['att'] / data['att'].mean())
# Created ranked dataframe based on mean statistics

gms_ranked = data.groupby(['fullname']).agg({

	'scaled_ypa': {'scaled_ypa':'mean'},

	'scaled_effc': {'scaled_effc':'mean'},

	'scaled_comppct': {'scaled_comppct':'mean'},

	'scaled_ptsacct': {'scaled_ptsacct':'mean'},

	'scaled_lps': {'scaled_lps':'mean'},

	'att': {'att':'mean'},

	'gm_score': {'gms_m': 'mean', 'gms_std': 'std'}})



gms_ranked.columns = gms_ranked.columns.droplevel(0)

gms_ranked.sort_values('gms_m', ascending=False, inplace = True)

print(gms_ranked.head(10))
# Histogram - Mean Game Score by Quarterback

rating_hist = sns.distplot(gms_ranked['gms_m'], bins = 25)

rating_hist.set_title('How the Ratings Stack Up')
# Bar Plot - Mean Game Score for Top Quarterbacks

rating_bar = sns.barplot(x = gms_ranked.index[0:10], y = 'gms_m', data = gms_ranked.head(10))

rating_bar.set_title('Top Performers')

rating_bar.axes.set_ylim(1.2,1.6)

rating_bar.axes.set_ylabel('Rating')

rating_bar.axes.set_xlabel('Quarterbacks')
# Box Plot - Mean Rating Per Game

ax_gms_box = plt.subplot(111)

ax_gms_box = sns.boxplot(data=gms_ranked[['scaled_lps','scaled_effc','scaled_ypa','scaled_comppct','scaled_ptsacct']])

ax_gms_box.set_title('Highest Variability in Efficiency, Points Accounted For, and Loss Per Sack')

ax_gms_box.set_xlabel('Features')

ax_gms_box.set_ylabel('Rating')
# Correlation Matrix - Rating and Features

corr = gms_ranked[['scaled_ypa','scaled_ptsacct','scaled_comppct','scaled_lps','scaled_effc','gms_m']].corr()

plt.figure(figsize = (12,12))

sns.heatmap(corr, vmax = 1, square = True)

print(corr['gms_m'])
stats = ['fullname','year','att', 'cmp', 'yds', 'td', 'intr', 'sack','loss', 'game_points']



seasons = data[stats].groupby(['fullname', 'year']).agg({

	'att': {'att_s': 'sum', 'att_m': 'mean'},

	'fullname': {'gms': 'count'},

	'cmp': {'cmp_s': 'sum', 'cmp_m': 'mean'},

	'yds': {'yds_s': 'sum', 'yds_m': 'mean'},

	'td': {'td_s': 'sum', 'td_m': 'mean'},

	'intr': {'intr_s': 'sum', 'intr_m': 'mean'},

	'sack': {'sack_s': 'sum', 'sack_m': 'mean'},

	'loss': {'loss_s': 'sum', 'loss_m': 'mean'},

	'game_points': {'gp_sum': 'sum'}})



seasons.columns = seasons.columns.droplevel(0)



seasons['ypa'] = seasons['yds_s'] / seasons['att_s']

seasons['lps'] = seasons['loss_s'] / seasons['sack_s']

seasons['comppct'] = seasons['cmp_s'] / seasons['att_s']

seasons['ptsacct'] = (seasons['td_s'] * 6) / seasons['gp_sum']

seasons['effc'] = 0

seasons.ix[(seasons.intr_s == 0) & (seasons.td_s != 0), 'effc'] = seasons['td_s']

seasons.ix[(seasons.intr_s == 0) & (seasons.td_s == 0), 'effc'] = season_defeffc

seasons.ix[(seasons.intr_s != 0), 'effc'] = seasons['td_s'] / seasons['intr_s']



print(seasons.head(10))
#Some season stats...

print('\nSEASON STATS\n','---------------')

print('Mean YPA: ', seasons['ypa'].mean())

print('Mean Completion Percentage: ', seasons['comppct'].mean())

print('Mean Points Accounted For: ', seasons['ptsacct'].mean())

print('Mean Loss Per Sack: ', seasons['lps'].mean())

print('Mean Efficiency: ', seasons['effc'].mean())
long_season = data.groupby(['fullname']).agg({

	'lg':'mean',

	'fullname': {'gms':'count'}})

long_season.columns = long_season.columns.droplevel(0)

long_season.rename(columns = {'mean':'lg'}, inplace = True)

long_season['scaled_lg'] = (long_season['lg'] / long_season['lg'].mean())

long_season.sort_values('scaled_lg', ascending = False, inplace = True)

print(long_season.head(15))
ha_gm = data.groupby(['fullname', 'home_away']).agg({

	'ypa': 'mean',

	'effc': 'mean',

	'comppct': 'mean',

	'ptsacct': 'mean',

	'lps': 'mean',

	'fullname': {'gms':'count'}})



ha_gm.columns = ha_gm.columns.droplevel(1)

ha_gm.rename(columns = {'fullname':'gms'}, inplace = True)

ha_cols = ['ypa','comppct','ptsacct','lps','effc'] #Columns for which we're testing the difference

ha_scores = pd.DataFrame(columns = ha_cols, index = ['fullname']) #Empty dataframe where we'll append scores
#Calculation: (Home mean - Away mean) / Total column mean

for qb in qbs:

	temp = {'fullname':qb}

	for col in ha_cols:

		temp[col] = (ha_gm[(ha_gm.index.get_level_values('fullname') == qb) & (ha_gm.index.get_level_values('home_away') == 'home')][col].values[0] - ha_gm[(ha_gm.index.get_level_values('fullname') == qb) & (ha_gm.index.get_level_values('home_away') == 'away')][col].values[0]) / data[col].mean()

	ha_scores = ha_scores.append(temp, ignore_index = True)



#Sum scores across columns to get a total and scale it relative to the mean

ha_scores.set_index(['fullname'], inplace = True)

ha_scores['ha'] = abs(ha_scores['ypa']) + abs(ha_scores['effc']) + abs(ha_scores['ptsacct']) + abs(ha_scores['lps']) + abs(ha_scores['comppct'])

ha_scores['scaled_ha'] = ha_scores['ha'] / ha_scores['ha'].mean()

ha_scores.sort_values(['scaled_ha'], ascending = True, inplace = True)

print(ha_scores.head(15))
games = data.groupby(['fullname'])['fullname'].count()

qbrat = data.groupby(['fullname'])['rate'].mean()



final = pd.concat([gms_ranked['gms_m'], long_season['scaled_lg'], ha_scores['scaled_ha'], games, qbrat], axis = 1, join = 'inner')

final.rename(columns = {'fullname':'gms'}, inplace = True)

final['nqbr'] = season_factor_weights['gms_m'] * final['gms_m'] + season_factor_weights['scaled_lg'] * final['scaled_lg'] + season_factor_weights['scaled_ha'] * final['scaled_ha']

final['nqbrover1'] = final['nqbr'] > 1

final.sort_values(['nqbr'], ascending = False, inplace = True)

print(final['nqbr'].head(20))
# Histogram - Mean Rating Per Game

totscore_hist = sns.distplot(final['nqbr'], bins = 25)

totscore_hist.set_title('How the Ratings Stack Up')
# Scatter - Games vs. Total Score

gms_totscore_scatter = sns.lmplot(

 	x = 'gms', y = 'nqbr',

 	data = final, hue = 'nqbrover1',

 	scatter_kws = {'marker': 'D', 's': 100})

plt.title('Ratings and Games Played')

plt.ylim(0.3,1.7)

plt.xlim(0,300)

plt.ylabel('Total Score')

plt.xlabel('Games Played')
rate_totscore_corr = final[['rate','nqbr']].corr()

print('Correlation - QB Rating vs. NQBR: ',rate_totscore_corr['rate']['nqbr'])



rate_totscore_scatter = sns.lmplot(

 	x = 'rate', y = 'nqbr',

 	data = final, scatter_kws = {'marker': 'D', 's': 100})

plt.title('NQBR vs. QB Rating')

plt.ylim(0.3,1.7)

plt.xlim(50,105)

plt.ylabel('NQBR')

plt.xlabel('QB Rating')
# Scatter - Game Score and NQBR by Games Played

gmsandnqbr_scatter, (gms_scatter, nqbr_scatter) = plt.subplots(nrows = 2, sharex = True)

sns.regplot(x = final['gms'], y = final['gms_m'], ax = gms_scatter)

sns.regplot(x = final['gms'], y = final['nqbr'], ax = nqbr_scatter)

plt.suptitle('NQBR Heavily Influenced by Mean Game Score')
# Box Plot - Mean Rating Per Game

totscore_box = plt.figure(figsize = (12,12))

totscore_box = sns.boxplot(data=final[['gms_m','scaled_lg','scaled_ha']])

totscore_box.set_title('Factor Influence on NQBR')

totscore_box.set_xlabel('Features')

totscore_box.set_ylabel('Rating')