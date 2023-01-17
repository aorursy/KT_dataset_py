# Data handling
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
from pandas.io.json import json_normalize

# save utils
import pickle

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import cufflinks as cf
cf.go_offline()

# ML
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from xgboost import plot_importance
train = pd.read_csv('../input/train_V2.csv').dropna() # One line is dropped
test = pd.read_csv('../input/test_V2.csv')
train.head()
print("There are %s matches in the training set and %s in the test set" % (train['matchId'].nunique(), test['matchId'].nunique()))
train_grouped = train[['matchType', 'matchId']]\
    .drop_duplicates().groupby('matchType').agg({'matchId': 'count'}).reset_index()

data = [go.Bar(x=train_grouped.matchType, y=train_grouped.matchId)]

layout = go.Layout(
    title='matchType training set distribution',
    xaxis=dict(
        title='matchType'
    ),
    yaxis=dict(
        title='Number of occurences'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
train_grouped = train[['Id', 'matchId']]\
    .groupby('matchId')\
    .agg({'Id': 'count'})\
    ['Id'].value_counts()\
    .reset_index()
                                                                 
data = [go.Bar(x=train_grouped['index'], y=train_grouped.Id)]

layout = go.Layout(
    title='Number of player per match distribution',
    xaxis=dict(
        title='Number of players'
    ),
    yaxis=dict(
        title='Number of occurences'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
train_grouped = train[['Id', 'matchId','matchType']]\
    .groupby(['matchId','matchType'], as_index=False)\
    .agg({'Id': 'count'})\
    .groupby(['matchType', 'Id'], as_index=False)\
    .agg({'matchId': 'count'})\
    .rename(columns={'Id': 'Nb_Players'})\
    .pivot(index='Nb_Players', columns='matchType', values='matchId')\
    .fillna(0)    

train_grouped.iplot(kind='bar', barmode='stack', title='Number of players distributions per matchType')   
train_grouped = train.assign(matchDurationMin = lambda x: np.floor(x.matchDuration / 60))[['matchDurationMin', 'matchId']]\
    .groupby('matchDurationMin', as_index = False)\
    .agg({'matchId': 'count'})\

data = [go.Bar(x=train_grouped.matchDurationMin, y=train_grouped.matchId)]

layout = go.Layout(
    title='Match duration in minute training set distribution',
    xaxis=dict(
        title='matchDuration in minutes'
    ),
    yaxis=dict(
        title='Number of occurences'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
train_grouped = train.assign(matchDurationMin = lambda x: np.floor(x.matchDuration / 60))[['matchType', 'matchDurationMin', 'matchId']]\
    .groupby(['matchType', 'matchDurationMin'], as_index = False)\
    .agg({'matchId': 'count'})\
    .pivot(index='matchDurationMin', columns='matchType', values='matchId')\
    .fillna(0)

train_grouped.iplot(kind='bar', barmode='stack', title='Match duration in minutes distribution per matchType')                   
def addNbPlayersFeature(df):
    df_res = df
    df_res = df_res.assign(nb_players_total=df_res.groupby('matchId')['Id'].transform('count'))
    df_res = df_res.assign(nb_players_team=df_res.groupby(['matchId', 'groupId'])['Id'].transform('count'))
    return df_res

train = addNbPlayersFeature(train)
def addNbKillsFeature(df):
    df_res = df.assign(nb_kills_team = df.groupby(['matchId', 'groupId'])['kills'].transform('sum'))
    return df_res

train = addNbKillsFeature(train)
def addKillPlaceQuantileInformation(df):
    df_res = df
    
    df_res['median_kill_place_team'] = df_res.groupby(['matchId', 'groupId'])['killPlace'].transform(np.median)
    df_res['median_kill_place_all'] = df_res.groupby('matchId')['killPlace'].transform(np.median)
    df_res['delta_median_kill_place_team_all'] = df_res.median_kill_place_all - df_res.median_kill_place_team
    df_res['max_kill_place_team'] = df_res.groupby(['matchId', 'groupId'])['killPlace'].transform(np.max)
    df_res['min_kill_place_team'] = df_res.groupby(['matchId', 'groupId'])['killPlace'].transform(np.min)

    return df_res
    
train = addKillPlaceQuantileInformation(train)
trf = train.groupby([pd.cut(train.median_kill_place_team, 25), pd.cut(train.median_kill_place_all, 25)]).winPlacePerc.mean().unstack()

data = [
    go.Contour(
        z=trf.values,
    )]

layout = go.Layout(
    title='mean winRankPct in the training set (color) given Median team kill Place and ',
    xaxis=dict(
        title='median killPlace team (/4)'
    ),
    yaxis=dict(
        title='median killPlace all (/4)'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
sample = train.sample(frac=0.0001)

# Create a trace
trace = go.Scatter(
    x = sample.median_kill_place_all - sample.median_kill_place_team,
    y = sample.winPlacePerc,
    text= ['winPlacePerc: %s' % s for s in sample.winPlacePerc],
    mode = 'markers'
)

data = [trace]

layout = go.Layout(
    title='winPlacePerc given difference in killPlace team/all (sample)',
    xaxis=dict(
        title='median killPlace all - median killPlace team'
    ),
    yaxis=dict(
        title='winPlacePerc'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
sample = train.sample(frac=0.0001)

# Create a trace
trace = go.Scatter(
    x = sample.median_kill_place_all - sample.killPlace,
    y = sample.winPlacePerc,
    text= ['winPlacePerc: %s' % s for s in sample.winPlacePerc],
    mode = 'markers',
    marker=dict(
        color=sample.nb_players_team
    )
)

data = [trace]

layout = go.Layout(
    title='winPlacePerc given difference in killPlace individual/all (sample)',
    xaxis=dict(
        title='median killPlace all - median killPlace individual'
    ),
    yaxis=dict(
        title='winPlacePerc'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
# /!\ quantiles 75 and 95 take time to compute
def addRankingQuantileInformation(df):
    df_res = df
    
    df_res['median_ranking_team'] = df_res.groupby(['matchId', 'groupId'])['rankPoints'].transform(np.median)
    #df_res['pct75_ranking_team'] = df_res.groupby(['matchId', 'groupId'])['rankPoints'].transform(lambda x:np.quantile(x, 0.75))
    #df_res['pct95_ranking_team'] = df_res.groupby(['matchId', 'groupId'])['rankPoints'].transform(lambda x:np.quantile(x, 0.95))
    
    df_res['median_ranking_all'] = df_res.groupby('matchId')['rankPoints'].transform(np.median)
    #df_res['pct75_ranking_all'] = df_res.groupby('matchId')['rankPoints'].transform(lambda x:np.quantile(x, 0.75))
    #df_res['pct95_ranking_all'] = df_res.groupby('matchId')['rankPoints'].transform(lambda x:np.quantile(x, 0.95))
    
    df_res['delta_median_ranking_team_all'] = df_res.median_ranking_all - df_res.median_ranking_team
    
    return df_res
    
train = addRankingQuantileInformation(train)
sample = train.sample(frac=0.0001)

# Create a trace
trace = go.Scatter(
    x = sample.median_ranking_team,
    y = sample.median_ranking_all,
    text= ['winPlacePerc: %s' % s for s in sample.winPlacePerc],
    mode = 'markers',
    marker=dict(
        size=7,
        color=10*sample.winPlacePerc
    )
)

data = [trace]
layout = go.Layout(
    title='winPlacePerc (bubble color) given median_ranking_team, median_ranking_all (sample)',
    xaxis=dict(
        title='median ranking team'
    ),
    yaxis=dict(
        title='median ranking all'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
sample = train.sample(frac=0.0001)

# Create a trace
trace = go.Scatter(
    x = sample.median_ranking_all - sample.median_ranking_team,
    y = sample.winPlacePerc,
    text= ['winPlacePerc: %s' % s for s in sample.winPlacePerc],
    mode = 'markers'
)

data = [trace]
layout = go.Layout(
    title='winPlacePerc given difference median_ranking_team, median_ranking_all (sample)',
    xaxis=dict(
        title='median ranking all - median ranking team'
    ),
    yaxis=dict(
        title='winPlacePerc'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
def addDamageDealtInformation(df):
    df_res = df
    
    df_res['mean_damage_dealt_team'] = df_res.groupby(['matchId', 'groupId'])['damageDealt'].transform(np.mean)
    df_res['max_damage_dealt_team'] = df_res.groupby(['matchId', 'groupId'])['damageDealt'].transform(np.max)
    df_res['min_damage_dealt_team'] = df_res.groupby(['matchId', 'groupId'])['damageDealt'].transform(np.min)
    
    df_res['mean_damage_dealt_all'] = df_res.groupby('matchId')['damageDealt'].transform(np.mean)
    df_res['max_damage_dealt_all'] = df_res.groupby('matchId')['damageDealt'].transform(np.max)
    df_res['min_damage_dealt_all'] = df_res.groupby('matchId')['damageDealt'].transform(np.min)
    
    return df_res
    
train = addDamageDealtInformation(train)
sample = train.sample(frac=0.0001)

# Create a trace
trace = go.Scatter(
    x = sample.mean_damage_dealt_all - sample.damageDealt,
    y = sample.winPlacePerc,
    text= ['winPlacePerc: %s' % s for s in sample.winPlacePerc],
    mode = 'markers'
)

data = [trace]

layout = go.Layout(
    title='winPlacePerc given difference mean_damageDealt_team, mean_damageDealt_all (sample)',
    xaxis=dict(
        title='mean damageDealt all - mean damageDealt team'
    ),
    yaxis=dict(
        title='winPlacePerc'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
def addDistanceWalked(df):
    df_res = df
    df_res['mean_distance_walked_team'] = df_res.groupby(['matchId', 'groupId'])['walkDistance'].transform(np.mean)
    df_res['max_distance_walked_team'] = df_res.groupby(['matchId', 'groupId'])['walkDistance'].transform(np.max)
    df_res['min_distance_walked_team'] = df_res.groupby(['matchId', 'groupId'])['walkDistance'].transform(np.min)
    
    df_res['mean_distance_walked_all'] = df_res.groupby(['matchId'])['walkDistance'].transform(np.mean)
    df_res['max_distance_walked_all'] = df_res.groupby(['matchId'])['walkDistance'].transform(np.max)
    df_res['min_distance_walked_all'] = df_res.groupby(['matchId'])['walkDistance'].transform(np.min)
    
    df_res['delta_mean_distance_walked_team_all'] = df_res.mean_distance_walked_team - df_res.mean_distance_walked_all
    
    return df_res

train = addDistanceWalked(train)
sample = train.sample(frac=0.0001)

# Create a trace
trace = go.Scatter(
    x = sample.mean_distance_walked_team - sample.mean_distance_walked_all,
    y = sample.winPlacePerc,
    text= ['winPlacePerc: %s' % s for s in sample.winPlacePerc],
    mode = 'markers',
    marker=dict(
        color=sample.nb_players_team
    )
)

data = [trace]
layout = go.Layout(
    title='winPlacePerc given difference mean_distance_walked_team, mean_distance_walked_all (sample)',
    xaxis=dict(
        title='mean_distance_walked_team - mean_distance_walked_all'
    ),
    yaxis=dict(
        title='winPlacePerc'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
def addWeaponsAcquired(df):
    df_res = df
    df_res['mean_weapons_acquired_team'] = df_res.groupby(['matchId', 'groupId'])['weaponsAcquired'].transform(np.mean)
    df_res['max_weapons_acquired_team'] = df_res.groupby(['matchId', 'groupId'])['weaponsAcquired'].transform(np.max)
    df_res['min_weapons_acquired_team'] = df_res.groupby(['matchId', 'groupId'])['weaponsAcquired'].transform(np.min)
    
    df_res['mean_weapons_acquired_all'] = df_res.groupby(['matchId'])['weaponsAcquired'].transform(np.mean)
    df_res['max_weapons_acquired_all'] = df_res.groupby(['matchId'])['weaponsAcquired'].transform(np.max)
    df_res['min_weapons_acquired_all'] = df_res.groupby(['matchId'])['weaponsAcquired'].transform(np.min)
    
    df_res['delta_mean_weapons_acquired_team_all'] = df_res.mean_weapons_acquired_team - df_res.mean_weapons_acquired_all
    
    return df_res

train = addWeaponsAcquired(train)
train_f = train[(train.mean_weapons_acquired_team>0) & (train.mean_weapons_acquired_team<10)]
train_f = train_f[(train_f.mean_weapons_acquired_all>0) & (train_f.mean_weapons_acquired_all<10)]

trf = train_f.groupby([pd.cut(train_f.mean_weapons_acquired_team, 5), pd.cut(train_f.mean_weapons_acquired_all, 5)]).winPlacePerc.mean().unstack()

data = [
    go.Contour(
        z=trf.values
    )]

layout = go.Layout(
    title='mean winRankPct in the training set (color) given mean acquired weapons team/all',
    xaxis=dict(
        title='mean acquired weapons team (/2)'
    ),
    yaxis=dict(
        title='mean acquired weapons all (/2)'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
sample = train.sample(frac=0.0001)

# Create a trace
trace = go.Scatter(
    x = sample.mean_weapons_acquired_all - sample.mean_weapons_acquired_team,
    y = sample.winPlacePerc,
    text= ['winPlacePerc: %s' % s for s in sample.winPlacePerc],
    mode = 'markers'
)

data = [trace]
layout = go.Layout(
    title='winPlacePerc given difference mean_acquiredWeapons all/team (sample)',
    xaxis=dict(
        title='mean_weapons_acquired_all - mean_weapons_acquired_team'
    ),
    yaxis=dict(
        title='winPlacePerc'
    )
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)
other_features = ['assists', 'boosts', 'DBNOs', 'headshotKills', 'heals',
            'revives', 'vehicleDestroys', 'killStreaks', 'roadKills']

def addOtherDeltaFeatures(df):
    df_res = df
    for f in other_features:
        df_res['mean_%s_team' % f] = df_res.groupby(['matchId', 'groupId'])[f].transform(np.mean)
        df_res['mean_%s_all' % f] = df_res.groupby('matchId')[f].transform(np.mean)
        df_res['delta_mean_%s_team_all' % f] = df_res['mean_%s_team' % f] - df_res['mean_%s_all' % f]
        
    return df_res

train = addOtherDeltaFeatures(train)
    
        
corr = train[['winPlacePerc',
       'delta_median_ranking_team_all',
       'delta_median_kill_place_team_all',
       'delta_mean_distance_walked_team_all',
      'delta_mean_weapons_acquired_team_all',
             'delta_mean_assists_team_all',
       'delta_mean_boosts_team_all', 'delta_mean_DBNOs_team_all',
       'delta_mean_headshotKills_team_all', 'delta_mean_heals_team_all',
       'delta_mean_revives_team_all', 'delta_mean_vehicleDestroys_team_all',
       'delta_mean_killStreaks_team_all', 'delta_mean_roadKills_team_all']].corr()

trace = go.Heatmap(z=corr.values,
                   x=corr.index,
                   y=corr.columns)
data=[trace]
py.iplot(data, filename='labelled-heatmap')
# Save train dataframe
train.to_pickle('train.pkl')

# Compute features and save test dataframe
test = addNbPlayersFeature(test)
test = addNbKillsFeature(test)
test = addRankingQuantileInformation(test)
test = addKillPlaceQuantileInformation(test)
test = addDistanceWalked(test)
test = addWeaponsAcquired(test)
test = addDamageDealtInformation(test)
test = addOtherDeltaFeatures(test)

test.to_pickle('test.pkl')
from sklearn.linear_model import LinearRegression, Ridge

features_selected = [
    'delta_median_kill_place_team_all',
    'delta_mean_distance_walked_team_all',
    'delta_mean_weapons_acquired_team_all',
    'delta_mean_boosts_team_all',
    'delta_mean_heals_team_all',
    'delta_mean_killStreaks_team_all',
    'winPlacePerc'
]
reg = LinearRegression()
#reg = Ridge(alpha = 100000)
train_linear = train[features_selected].dropna()
test_linear = test[[f for f in features_selected if f!='winPlacePerc'] + ['Id']]
Y_train = train_linear.winPlacePerc
X_train = train_linear.drop(columns=['winPlacePerc'], axis=1)

reg = reg.fit(X_train, Y_train)
first_pred = reg.predict(X_train)
# With a linear regression we could expect ~0.08MAE
np.mean(abs(first_pred-Y_train))
train = pd.read_pickle('train.pkl')
test = pd.read_pickle('test.pkl')

test.columns
# Select features
features_selected = [
    # - - - Dataset features 
    #'Id',
    #'groupId',
    #'matchId',
    'assists',
    'boosts',
    #'damageDealt',
    #'DBNOs',
    #'headshotKills',
    'heals',
    'killPlace',
    'killPoints',
    'kills',
    'killStreaks',
    #'longestKill',
    'matchDuration',
    #'matchType',
    #'maxPlace',
    'numGroups',
    'rankPoints',
    #'revives',
    #'rideDistance',
    #'roadKills',
    #'swimDistance',
    'teamKills',
    #'vehicleDestroys',
    #'walkDistance',
    #'weaponsAcquired',
    'winPoints',
    'winPlacePerc',
    # - - - Computed features 
    'nb_players_total',
    'nb_players_team',
    'nb_kills_team',
    'median_ranking_team',
    'median_ranking_all',
    'median_kill_place_team',
    'delta_median_ranking_team_all',
    'delta_median_kill_place_team_all',
    'mean_damage_dealt_team',
    'max_damage_dealt_team',
    'min_damage_dealt_team',
    'mean_damage_dealt_all',
    'max_damage_dealt_all',
    'min_damage_dealt_all',
    #'delta_mean_distance_walked_team_all',
    'mean_distance_walked_team',
    'max_distance_walked_team',
    'min_distance_walked_team',
    'mean_distance_walked_all',
    'max_distance_walked_all',
    'min_distance_walked_all',
    #'delta_mean_weapons_acquired_team_all',
    'max_weapons_acquired_team',
    'min_weapons_acquired_team',
    'mean_weapons_acquired_team',
    'max_weapons_acquired_all',
    'min_weapons_acquired_all',
    'mean_weapons_acquired_all',
    #'linear_pred',
    #'delta_mean_assists_team_all',
    'mean_assists_all',
    'mean_assists_team',
    #'delta_mean_boosts_team_all',
    'mean_boosts_all',
    'mean_boosts_team',
    #'delta_mean_DBNOs_team_all',
    'mean_DBNOs_all',
    'mean_DBNOs_team',
    #'delta_mean_headshotKills_team_all',
    'delta_mean_heals_team_all',
    #'delta_mean_revives_team_all',
    #'delta_mean_vehicleDestroys_team_all',
    'delta_mean_killStreaks_team_all',
    'delta_mean_roadKills_team_all',
    #'mean_roadKills_team',
    #'mean_roadKills_all'
]

train = train[features_selected].dropna()
test = test[[f for f in features_selected if f!='winPlacePerc'] + ['Id']]

train, valid = train_test_split(train, test_size=0.2)
Y_train = train.winPlacePerc
X_train = train.drop(columns=['winPlacePerc'], axis=1)

Y_valid = valid.winPlacePerc
X_valid = valid.drop(columns=['winPlacePerc'], axis=1)
X_train.info()

from xgboost import XGBRegressor
from xgboost import plot_importance
import time
ts = time.time()

model = XGBRegressor(
    max_depth=6,
    n_estimators=800,
    min_child_weight=100, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="mae", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))

Y_pred = model.predict(test).clip(0, 1)
Y_pred


submission = pd.DataFrame({
    "ID": test.Id, 
    "winPlacePerc": Y_pred
})
submission.to_csv('xgb_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
submission
