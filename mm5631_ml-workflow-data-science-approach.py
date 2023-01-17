import os
import sys

import pandas as pd
import numpy as np
import datetime

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor

import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import  init_notebook_mode, iplot
init_notebook_mode()

import shap

DATA_DIR = '../input'
RANDOM_STATE = 212

pd.options.display.max_columns = 60
pd.options.display.float_format = '{0:.2f}'.format

sns.set_style('darkgrid') 

%config InlineBackend.figure_format = 'retina'
%matplotlib inline
def shape(df):
    return '{:,} rows - {:,} columns'.format(df.shape[0], df.shape[1])
data = pd.read_csv('{}/{}'.format(DATA_DIR, 'train.csv'))
data.columns = data.columns.str.lower()
shape(data)
data.head()
def plot_hist(x, title):
    
    fig, ax = plt.subplots(figsize=(13,7))
    formatter = plt.FuncFormatter(lambda x, y: '{:,.2f}'.format(x))
    
    ax.yaxis.set_major_formatter(formatter=formatter)
    ax.xaxis.set_major_formatter(formatter=formatter)

    ax.set_title(title)
    sns.distplot(x, bins=50, kde=False, ax=ax);
print('The average winning percentile is {:.3f}, the median is {:.3f}'.format(data.winplaceperc.mean(), data.winplaceperc.median()))
plot_hist(data.winplaceperc, title='Histogram of winning percentiles')
data = data.assign(match_mean=data.groupby('matchid').winplaceperc.transform('mean'))
data = data.assign(match_median = data.groupby('matchid').winplaceperc.transform('median'))
print('The average match winning percentile is {:.2f}, the median is {:,.2f}'.format(data.winplaceperc.mean(), data.winplaceperc.median()))
plot_hist(data.match_mean, title='Histogram of average match winning percentiles');
plot_hist(data.match_median, title='Histogram of median match winning percentiles');
data = data.assign(team_size=data.groupby('groupid').groupid.transform('count'))
data = data.assign(max_team_size=data.groupby('matchid').team_size.transform('max'))
data= data.assign(match_size=data.groupby('matchid').id.transform('nunique'))
plot_hist(data.match_size, title='Distribution of players per game')
print('The largest team has {} team members'.format(data.max_team_size.max()))
plot_hist(data.team_size, title='Histogram of team sizes')
plt.xlim(0,12);
data[data.max_team_size == 2].team_size.value_counts()
plot_hist(data.max_team_size, title='Distribution of maximum team size')
plt.xlim(0,20);
data =  data.assign(team_indicator = data.team_size.apply(lambda x: 5 if x>= 5 else x))
data = pd.get_dummies(data, columns=['team_indicator'])
dummy_cols = ['team_indicator_{}'.format(i) for i in np.arange(1,6)]
data[dummy_cols] = data.groupby('matchid')[dummy_cols].transform('mean')
data.head()
plot_hist(data[data.max_team_size==2].team_indicator_1, title='Distribution of solo teams density where maximum team size is 2')
data.loc[data.team_indicator_1 >= 0.7, 'game_mode'] = 'solo'
data.loc[data.team_indicator_2 >= 0.6, 'game_mode'] = 'duo'
data.loc[(data.team_indicator_3 + data.team_indicator_4) >= 0.5, 'game_mode'] = 'squad'

data.game_mode = np.where((data.team_indicator_5 >= 0.2), 'custom', data.game_mode)
data.game_mode = data.game_mode.fillna('custom')
data[dummy_cols+['game_mode']].sample(15)
print('The average winning percentile for regular games is {:.3f}, the median is {:.3f}'.format(data[data.game_mode!='custom'].winplaceperc.mean(), data[data.game_mode!='custom'].winplaceperc.median()))
print('The average winning percentile for custom games is {:.3f}, the median is {:.3f}'.format(data[data.game_mode=='custom'].winplaceperc.mean(), data[data.game_mode=='custom'].winplaceperc.median()))

plot_hist(data[data.game_mode != 'custom'].winplaceperc, title = 'Histogram of winning percentiles scores for regular games')
plot_hist(data[data.game_mode == 'custom'].winplaceperc, title = 'Histogram of winning percentiles scores for custom games')
data['max_possible_kills'] = data.match_size - data.team_size
data['total_distance'] = data.ridedistance + data.swimdistance + data.walkdistance
data['total_items_acquired'] = data.boosts + data.heals + data.weaponsacquired
data['items_per_distance'] =  data.total_items_acquired/data.total_distance
data['items_per_distance'] =  data.total_items_acquired/data.total_distance
data['kills_per_distance'] = data.kills/data.total_distance
data['knocked_per_distance'] = data.dbnos/data.total_distance
data['damage_per_distance'] = data.damagedealt/data.total_distance
data['headshot_kill_rate'] = data.headshotkills/data.kills
data['max_kills_by_team'] = data.groupby('groupid').kills.transform('max')
data['total_team_damage'] = data.groupby('groupid').damagedealt.transform('sum')
data['total_team_kills'] =  data.groupby('groupid').kills.transform('sum')
data['total_team_items'] = data.groupby('groupid').total_items_acquired.transform('sum')
data['pct_killed'] = data.kills/data.max_possible_kills
data['pct_knocked'] = data.dbnos/data.max_possible_kills
data['pct_team_killed'] = data.total_team_kills/data.max_possible_kills
data['team_kill_points'] = data.groupby('groupid').killpoints.transform('sum')
data['team_kill_rank'] = data.groupby('groupid').killplace.transform('mean')
data['max_kills_match'] = data.groupby('matchid').kills.transform('max')
data['total_kills_match'] = data.groupby('matchid').kills.transform('sum')
data['total_distance_match'] = data.groupby('matchid').total_distance.sum()
data['map_has_sea'] =  data.groupby('matchid').swimdistance.transform('sum').apply(lambda x: 1 if x>0 else 0)
data.fillna(0, inplace=True)
def plot_interactions(df, feature_list, hue_labels=None, sample_size=10000):
    
    '''
    Target to decile should be first
    '''
    sample_df = df.sample(sample_size)
    sample_df.team_size = sample_df.team_size.apply(lambda x: 5 if x>= 5 else x)
    
    colors = pd.qcut(sample_df[feature_list[0]], q=10, labels=np.arange(1,11)).astype(int)
    colorscale = 'RdBu'
    
    trace = [go.Parcoords(
        line = dict(color=colors, colorscale = colorscale),
        dimensions = list([dict(range = [np.round(sample_df[i].quantile(0.01)*0.9, decimals=1),
                                         np.round(sample_df[i].quantile(0.99)*1.1, decimals=1)],
                                label = str(i),
                                values = sample_df[i]) for i in feature_list]))]

    fig = go.Figure(data=trace)
    iplot(fig)
features = ['winplaceperc', 'walkdistance', 'damagedealt', 'boosts', 'total_items_acquired', 'revives']
plot_interactions(df=data, feature_list=features)
EXCLUDE_COLS = ['id', 'match_mean', 'match_median', 'team_indicator_5', 'game_mode']
CATEGORICAL_COLS = ['matchid', 'groupid']
TARGET = 'winplaceperc'
TRAIN_SIZE = 0.9
EARLY_STOP_ROUNDS = 10
df = data[data.game_mode != 'custom'].drop(EXCLUDE_COLS, axis=1)
df[CATEGORICAL_COLS] =  df[CATEGORICAL_COLS].astype('category')
shape(df)
def train_validation(df, train_size=TRAIN_SIZE):
    
    unique_games = df.matchid.unique()
    train_index = round(int(unique_games.shape[0]*train_size))
    
    np.random.shuffle(unique_games)
    
    train_id = unique_games[:train_index]
    validation_id = unique_games[train_index:]
    
    train = df[df.matchid.isin(train_id)]
    validation = df[df.matchid.isin(validation_id)]
    
    return train, validation
    
train, validation = train_validation(df)
train_weights = (1/train.team_size)
validation_weights = (1/validation.team_size)
X_train = train.drop(TARGET,axis=1)
X_test = validation.drop(TARGET, axis=1)

y_train = train[TARGET]
y_test = validation[TARGET]

shape(X_train), shape(X_test)
time_0 = datetime.datetime.now()

lgbm = LGBMRegressor(objective='mae', n_estimators=250,  
                     learning_rate=0.3, num_leaves=200, 
                     n_jobs=-1,  random_state=RANDOM_STATE, verbose=0)

lgbm.fit(X_train, y_train, sample_weight=train_weights,
         eval_set=[(X_test, y_test)], eval_sample_weight=[validation_weights], 
         eval_metric='mae', early_stopping_rounds=EARLY_STOP_ROUNDS, 
         verbose=0)

time_1  = datetime.datetime.now()

print('Training took {} seconds. Best iteration is {}'.format((time_1 - time_0).seconds, lgbm.best_iteration_))
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, lgbm.predict(X_test, num_iteration=lgbm.best_iteration_), sample_weight=validation_weights)))
print('R2 score is {:.2%}'.format(r2_score(y_test, lgbm.predict(X_test, num_iteration=lgbm.best_iteration_), sample_weight=validation_weights)))
def plot_training(lgbm):
    
    fig, ax = plt.subplots(figsize=(13,7))
    losses = lgbm.evals_result_['valid_0']['l1']
    ax.set_ylim(np.max(losses), 0)
    ax.set_xlim(0,100)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Mean Asbolute Error')
    ax.set_title('Evolution of MAE over training iterations')
    ax.plot(losses, color='grey');
    
plot_training(lgbm)
results = validation.copy()
results = results.assign(predicted_player_rank=lgbm.predict(X_test, num_iteration=lgbm.best_iteration_))
print('The minimum predicted ranking is {}, the maximum is {}'.format(results.predicted_player_rank.min(), results.predicted_player_rank.max()))
results.predicted_player_rank = results.predicted_player_rank.clip(0, 1)
print('The minimum predicted ranking is {}, the maximum is {}'.format(results.predicted_player_rank.min(), results.predicted_player_rank.max()))
print('R2 score is {:.2%}'.format(r2_score(y_test, results.predicted_player_rank, sample_weight=validation_weights)))
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, results.predicted_player_rank, sample_weight=validation_weights)))
results = results.assign(predicted_team_rank_max=results.groupby('groupid').predicted_player_rank.transform('max'))
results = results.assign(predicted_team_rank_mean=results.groupby('groupid').predicted_player_rank.transform('mean'))

print('Using team maximum predicted ranking:')
print('R2 score is {:.2%}'.format(r2_score(y_test, results.predicted_team_rank_max.clip(0, 1), sample_weight=validation_weights)))
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, results.predicted_team_rank_max, sample_weight=validation_weights)))

print('\nUsing team average predicted ranking:')
print('R2 score is {:.2%}'.format(r2_score(y_test, results.predicted_team_rank_mean.clip(0, 1), sample_weight=validation_weights)))
print('Mean Absolute Error is {:.5f}'.format(mean_absolute_error(y_test, results.predicted_team_rank_mean, sample_weight=validation_weights)))

sns.jointplot(y_test, results.predicted_team_rank_mean,
              kind='reg', height=12,
              xlim=(-0.1, 1.1), ylim=(-0.1, 1.1),
              color='darkred', scatter_kws={'edgecolor':'w'}, line_kws={'color':'black'});
plt.title('Actual Ranking vs Predicted Ranking');
shap.initjs()

SAMPLE_SIZE = 10000
SAMPLE_INDEX = np.random.randint(0, X_test.shape[0], SAMPLE_SIZE)

X = X_test.iloc[SAMPLE_INDEX]

explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_type='bar', color='lightblue')
shap.summary_plot(shap_values, X)
interactions = ['assists', 'boosts', 'damagedealt', 'heals', 'longestkill', 'walkdistance', 'revives']
features = ['team_kill_rank'] * len(interactions)

for i, j in zip(features, interactions):
    shap.dependence_plot(i, shap_values, X, interaction_index=j);
