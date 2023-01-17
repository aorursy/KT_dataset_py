# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Any results you write to the current directory are saved as output.
# load data

player_stats = pd.read_csv('../input/game_skater_stats.csv')

team_stats = pd.read_csv('../input/game_teams_stats.csv')

game = pd.read_csv('../input/game.csv')

player_info = pd.read_csv('../input/player_info.csv')
# look at first few rows

player_stats.head()
# info on player_stats

player_stats.info()
# summary statistics

player_stats.describe().T
def draw_4_distplots(var1, var2, var3, var4, data):

    """

    Displays 4 distplots on a 2x2 grid

    """

    fig, axes = plt.subplots(2,2,figsize=(20,12))



    sns.distplot(data[var1], ax=axes[0,0])

    axes[0,0].set_title(var1 + ' distribution')

    sns.distplot(data[var2], ax=axes[0,1])

    axes[0,1].set_title(var2 + ' distribution')

    sns.distplot(data[var3], ax=axes[1,0])

    axes[1,0].set_title(var3 + ' distribution')

    sns.distplot(data[var4], ax=axes[1,1])

    axes[1,1].set_title(var4 + ' distribution')



    plt.tight_layout()

    plt.show()
# look at distribution of few of the variables

draw_4_distplots('shots', 'timeOnIce', 'goals', 'assists', player_stats)
# heatmap of correlations between variables in the data

fig, ax = plt.subplots(figsize=(24,16))

sns.heatmap(player_stats.corr(),

            xticklabels=player_stats.corr().columns,

            yticklabels=player_stats.corr().columns,

            annot=True, fmt='.2f', ax=ax, cmap='bone')

plt.show()
# drop irrelevant columns

player_stats.drop(['hits', 'penaltyMinutes', 'faceOffWins',

                   'faceoffTaken', 'giveaways', 'shortHandedGoals',

                   'shortHandedAssists', 'blocked', 'evenTimeOnIce',

                   'shortHandedTimeOnIce', 'powerPlayTimeOnIce'],

                    axis=1, inplace=True)
def draw_regplot(x_data,y_data,x_label,y_label,title):

    """

    Display scatterplot with regression line between x_data and y_data

    """ 

    fig, ax = plt.subplots(figsize=(18,12))

    sns.regplot(x=x_data, y=y_data, color='b', ax=ax)

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)

    plt.show()
# take averages of the variables in the data and display relationship between shots and goals

avgs = player_stats.groupby('player_id').mean()

draw_regplot(avgs.goals, avgs.shots,

             'Average goals', 'Average shots',

             'Average goals by average shots')
# relationship between avg ice time and avg shots

draw_regplot(avgs.timeOnIce, avgs.shots,

             'Average ice time', 'Average shots',

             'Average ice time by average shots')
# get primary position column from player_info table

player_stats = pd.merge(player_stats, player_info[['player_id','primaryPosition']],

                        on='player_id', how='left')
def draw_barplot(x_data,y_data,x_label,y_label,title,c):

    """

    Display barplot of x_dta and y_data

    """

    fig, ax = plt.subplots(figsize=(16,10))                                        

    sns.barplot(x=x_data, y=y_data, palette=c, ax=ax) 

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.title(title)

    plt.show()
# plot average shots/game by different player positions

draw_barplot(player_stats.primaryPosition, player_stats.shots,

             'Position', 'Mean shots', 'Average shots taken by position', 'Blues')
# look at head of team_stats data

team_stats.head()
# the columns I'm interested in are home/away and team shots

# there are 2 entries for each game, 1 for each team, so we have to merge on both game_id and team_id

player_stats = pd.merge(player_stats, team_stats[['game_id', 'team_id', 'HoA', 'shots']],

                        on=['game_id', 'team_id'], how='left', suffixes=('','_team'))
# plot average shots taken in home and away games

draw_barplot(player_stats.HoA, player_stats.shots,

             'Home/Away', 'Mean shots', 'Average shots taken home/away', 'Blues')
# look at first rows of game table 

game.head()
# merge player table with relevant columns in game table

player_stats = pd.merge(player_stats,

                        game[['game_id','season', 'date_time', 'away_team_id', 'home_team_id']],

                        on='game_id', how='left')
player_stats.date_time = pd.to_datetime(player_stats.date_time)

player_stats.date_time = player_stats.date_time.dt.normalize()

player_stats.set_index('date_time', inplace=True)

player_stats.sort_index(inplace=True)
# create a new column in the dataframe with opponent_id for each players entry

def get_opponent(df):

    if df['HoA'] == 'away':

        return df['home_team_id']

    else:

        return df['away_team_id']

    

player_stats['opponent_id'] = player_stats.apply(get_opponent, axis=1)

player_stats.drop(['away_team_id','home_team_id'], axis=1, inplace=True)
# calculate how many shots each opponent allows on average

shots_allowed = pd.DataFrame(player_stats.groupby('opponent_id', as_index=False)['shots_team'].mean())
draw_barplot(shots_allowed.opponent_id, shots_allowed.shots_team,

             'Team', 'Average shots allowed', 'Average shots allowed/game', 'husl')
# select 4 different players and display how they shoot against different teams on average

player1 = player_stats.loc[(player_stats.player_id == 8471724) & (player_stats.season == 20172018)]

player2 = player_stats.loc[(player_stats.player_id == 8474190) & (player_stats.season == 20172018)]

player3 = player_stats.loc[(player_stats.player_id == 8473512) & (player_stats.season == 20172018)]

player4 = player_stats.loc[(player_stats.player_id == 8475167) & (player_stats.season == 20172018)]



fig, axes = plt.subplots(2,2,figsize=(26,18))



sns.barplot(player1.opponent_id, player1.shots, ax=axes[0,0])

axes[0,0].set_title('Player 1')

sns.barplot(player2.opponent_id, player2.shots, ax=axes[0,1])

axes[0,1].set_title('Player 2')

sns.barplot(player3.opponent_id, player3.shots, ax=axes[1,0])

axes[1,0].set_title('Player 3')

sns.barplot(player4.opponent_id, player4.shots, ax=axes[1,1])

axes[1,1].set_title('Player 4')



plt.tight_layout()

plt.show()
draw_barplot(player_stats.shots.value_counts().sort_index().index,

             player_stats.shots.value_counts().sort_index().values,

             'Number of shots', 'Count', 'Count of shots taken/game', 'husl')
# filter out player who did not play at least 10 games

rowsBefore = player_stats.shape[0]

player_stats = player_stats.groupby('player_id').filter(lambda x: len(x) >= 10)

rowsAfter = player_stats.shape[0]

print('{} rows filtered out'.format(rowsBefore - rowsAfter))
# drop rows where a player has more than one entry on the same day, keep the first one

def drop_dup(df):

    grp = df.groupby(['player_id', 'date_time'])

    dropped = grp.first()

    dropped.reset_index(level=0, inplace=True)

    dropped.sort_index(inplace=True)

    return dropped

rowsBefore = player_stats.shape[0]

player_stats = drop_dup(player_stats)

rowsAfter = player_stats.shape[0]

print('{} rows filtered out'.format(rowsBefore - rowsAfter))
# take the average number of shots for each player each season

mean_season = pd.DataFrame(player_stats.groupby(['player_id', 'season'])['shots'].mean())



player_stats_1 = pd.merge(player_stats, mean_season, on=['player_id', 'season'], how='left', suffixes=('', '_mean_season'))

player_stats_1.index = player_stats.index
# take the standard deviation of shots for each player each season

std_season = pd.DataFrame(player_stats.groupby(['player_id', 'season'])['shots'].std())



player_stats_1 = pd.merge(player_stats_1, std_season, on=['player_id', 'season'], how='left', suffixes=('', '_std_season'))

player_stats_1.index = player_stats.index
# get the average number of shots against each opponent for each player

mean_v_opponent = pd.DataFrame(player_stats_1.groupby(['player_id', 'opponent_id'])['shots'].mean())



player_stats_1 = pd.merge(player_stats_1, mean_v_opponent, on=['player_id', 'opponent_id'], how='left', suffixes=('', '_mean_v_opp'))

player_stats_1.index = player_stats.index
# calculate the ratio of games a player has taken more than 2 shots

temp = pd.DataFrame()

for plyr in player_stats_1['player_id'].unique():

    for szn in player_stats_1['season'].unique():

        l = player_stats_1[['player_id', 'shots', 'season']].loc[(player_stats_1['player_id'] == plyr) & (player_stats_1['season'] == szn)]

        if (len(l)>0):

            games_o_2s = (len(l.loc[l['shots'] > 2]))/(len(l))

        else:

            games_o_2s = 0

        temp = temp.append([[plyr, games_o_2s, szn]])



temp.rename(columns={0:'player_id', 1:'shots_over2', 2:'season'}, inplace=True)



player_stats_1 = pd.merge(player_stats_1, temp, on=['player_id', 'season'], how='left')

player_stats_1.index = player_stats.index
# calculate sum of shots over the previous 5 games

def shots_last5(df):

    df['shots_shifted'] = df.groupby('player_id')['shots'].shift() # need to shift the data 1 step to exclude the 'current' game

    grouped = pd.DataFrame(df.groupby('player_id')['shots_shifted'].rolling(5).sum())

    grouped.reset_index(level=0, inplace=True)

    grouped['shots_shifted'] = grouped.groupby('player_id')['shots_shifted'].transform(lambda x: x.fillna(x.mean()))

    return grouped



s_last5 = shots_last5(player_stats_1)

player_stats_1 = pd.merge(player_stats_1, s_last5, on=['date_time', 'player_id'], suffixes=('', '_last5'))

player_stats_1.drop('shots_shifted', axis=1, inplace=True)

player_stats_1.rename(columns={'shots_shifted_last5':'shots_last5'}, inplace=True)
# calculate the sum of shots in the last 5 games against each opponent

def shots_last5_v_opp(df):

    df['shots_shifted'] = df.groupby(['player_id','opponent_id'])['shots'].shift() 

    grouped = pd.DataFrame(df.groupby(['player_id','opponent_id'])['shots_shifted'].rolling(5).sum())

    grouped.reset_index(level=[0,1], inplace=True)

    grouped['shots_shifted'] = grouped.groupby(['player_id','opponent_id'])['shots_shifted'].transform(lambda x: x.fillna(x.mean()))

    return grouped



s_last5opp = shots_last5_v_opp(player_stats_1)

player_stats_1 = pd.merge(player_stats_1, s_last5opp, on=['date_time', 'player_id', 'opponent_id'], suffixes=('', '_last5_vs_opp'))

player_stats_1.drop('shots_shifted', axis=1, inplace=True)

player_stats_1.rename(columns={'shots_shifted_last5_vs_opp':'shots_last5_vs_opp'}, inplace=True)

player_stats_1.fillna(player_stats_1['shots_last5_vs_opp'].mean(), inplace=True)
# take the average ice time over the last 5 games

def ice_mean_last5(df):

    df['icetime_shifted'] = df.groupby('player_id')['timeOnIce'].shift()

    grouped = pd.DataFrame(df.groupby('player_id')['icetime_shifted'].rolling(5).mean())

    grouped.reset_index(level=0, inplace=True)

    grouped['icetime_shifted'] = grouped.groupby('player_id')['icetime_shifted'].transform(lambda x: x.fillna(x.mean()))

    return grouped



ice_last5 = ice_mean_last5(player_stats_1)

player_stats_1 = pd.merge(player_stats_1, ice_last5, on=['date_time', 'player_id'], suffixes=('', '_last5'))

player_stats_1.drop('icetime_shifted', axis=1, inplace=True)

player_stats_1.rename(columns={'icetime_shifted_last5':'mean_ice_last5'}, inplace=True)
draw_4_distplots('shots_mean_season',

                'shots_std_season',

                'shots_over2',

                'mean_ice_last5',

                player_stats_1)
# missing data check

player_stats_1.isnull().any()
# downsample the dataset to get a balance between classes

from sklearn.utils import resample



majority = player_stats_1[player_stats_1.shots<=2]

minority = player_stats_1[player_stats_1.shots>2]



majority_downsampled = resample(majority, replace=False,

                                n_samples=len(minority), random_state=1)

player_stats_downsampled = pd.concat([majority_downsampled, minority])

player_stats_downsampled.sort_index(inplace=True)

player_stats_downsampled = player_stats_downsampled.groupby('player_id').filter(lambda x: len(x) >= 10)
before = player_stats_1.shots.map(lambda x: 1 if x>2 else 0).value_counts().sort_index()

after = player_stats_downsampled.shots.map(lambda x: 1 if x>2 else 0).value_counts().sort_index()

before.rename(index={0: 'Under', 1:'Over'}, inplace=True)

after.rename(index={0: 'Under', 1:'Over'}, inplace=True)



fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,8))



ax1.set(title='Before downsample',

        xlabel='Shots under and over 2',

        ylabel='No of samples')

sns.barplot(before.index, before.values, palette='Blues', ax=ax1)

ax2.set(title='After downsample',

        xlabel='Shots under and over 2',

        ylabel='No of samples')

sns.barplot(after.index, after.values, palette='Blues', ax=ax2)



plt.tight_layout()

plt.show()
# look at how the first rows of the current dataset

player_stats_downsampled.head()
# grab the target column

Y = player_stats_downsampled['shots'].map(lambda x: 1 if x>2 else 0).values
# convert the categorical columns to numerical

player_stats_downsampled.primaryPosition = player_stats_downsampled.primaryPosition.map(lambda x: 0 if x == 'D' else 1) # 1 if forward 0 if defense

player_stats_downsampled.HoA = player_stats_downsampled.HoA.map(lambda x: 1 if x == 'home' else 0) # 1 if home 0 if away
# grab the features to use for training the model

features = player_stats_downsampled[['primaryPosition', 'HoA', 'shots_mean_season',

                                     'shots_std_season', 'shots_mean_v_opp', 'shots_over2',

                                     'shots_last5', 'shots_last5_vs_opp', 'mean_ice_last5']]
# first few rows of the features

features.head()
# standardize the data

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



features_scaled = scaler.fit_transform(features)
# split up the data with the last 20% (time wise) as testing portion

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features_scaled, Y,

                                                    test_size=.2, shuffle=False)
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV



tscv = TimeSeriesSplit(n_splits=5)

logit = LogisticRegression(random_state=1, class_weight='balanced')



params = {'C': np.logspace(-2,2,10),

          'solver': ['liblinear', 'lbfgs', 'sag']}



logit_grid_searcher = GridSearchCV(estimator=logit,

                                   param_grid=params,

                                   cv=tscv,

                                   scoring='roc_auc',

                                   verbose=1)

logit_grid_searcher.fit(X_train, y_train)



print('Best params: {}'.format(logit_grid_searcher.best_params_))
from sklearn.metrics import accuracy_score, confusion_matrix



preds = logit_grid_searcher.predict(X_test)

print('Prediction accuracy on test data: {}%'.format(np.round(accuracy_score(y_test,preds)*100, 2)))



cf = confusion_matrix(y_test, preds)

fig,ax=plt.subplots(figsize=(10,8))

sns.heatmap(cf, annot=True, cmap='Blues', ax=ax, fmt='d')

ax.set_title('Confusion Matrix')

ax.set_ylabel('True Label')

ax.set_xlabel('Predicted Label')

plt.show()
np.round(pd.DataFrame(logit_grid_searcher.predict_proba(X_test),

                      columns=['UNDER 2.5', 'OVER 2.5']).apply(lambda x: 1/x).head(10), 2)