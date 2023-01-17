import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)
import gc
from random import sample
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.preprocessing import LabelEncoder

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

df_train= pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
df_test= pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
df_train['belongs']= 'train'
df_test['belongs']= 'test'
df= pd.concat([df_train.drop('winPlacePerc', axis= 1), df_test], axis= 0, ignore_index= True)
df['killRate']= df['kills']/ df['matchDuration']
df['DBNORate']= df['DBNOs']/ df['matchDuration']
df['entryCount']= 1
df['total_players_match']= df.groupby(['matchId'])['entryCount'].transform(np.sum)
df['total_players_group']= df.groupby(['groupId'])['entryCount'].transform(np.sum)
df['killPlacePerc']= (df['killPlace']/ df['total_players_match'])
#df.drop('killPlace', axis= 1, inplace= True)
df.loc[df['killPoints']== 0, 'killPoints']= 1
df['maxKillPointsMatch']= df.groupby(['matchId'])['killPoints'].transform(np.max)
df['maxKillPointsGroup']= df.groupby(['groupId'])['killPoints'].transform(np.max)
df['ratioMatchKillPoints']= df['killPoints']/ df['maxKillPointsMatch']
df['ratioGroupKillPoints']= df['killPoints']/ df['maxKillPointsGroup']
df['killPointsBuckets']= pd.cut(df['killPoints'], bins= [0, 220, 450, 650, 870, 1100, 1300, 1500, 1750, 1900, 2200], labels= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], include_lowest= True)
df['killPointsBuckets']= df['killPointsBuckets'].astype(np.int8)
df.loc[df['winPoints']== 0, 'winPoints']= 1
df['maxWinPointsMatch']= df.groupby(['matchId'])['winPoints'].transform(np.max)
df['maxWinPointsGroup']= df.groupby(['groupId'])['winPoints'].transform(np.max)
df['ratioMatchWinPoints']= df['winPoints']/ df['maxKillPointsMatch']
df['ratioGroupWinPoints']= df['winPoints']/ df['maxKillPointsGroup']
plt.hist(df['winPoints'], bins= 10)
df['winPointsBuckets']= pd.cut(df['winPoints'], bins= [0, 200, 403, 604, 806, 1007, 1208, 1409, 1610, 1811, 2020], labels= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], include_lowest= True)
df['winPointsBuckets']= df['winPointsBuckets'].astype(np.int8)
#df.drop(['maxKillPointsMatch', 'maxKillPointsGroup', 'maxWinPointsMatch', 'maxWinPointsGroup'], axis= 1, inplace= True)
df['killPointsSumMatch']= df.groupby(['matchId'])['killPoints'].transform(np.sum)
df['killPointsSumGroup']= df.groupby(['groupId'])['killPoints'].transform(np.sum)
df['ratioKillPointsGroupAndMatch']= df['killPointsSumGroup']/ df['killPointsSumMatch'] 
df['avgKillPointsGroup']= df.groupby(['matchId'])['killPoints'].transform(np.mean)
df['avgKillPointsMatch']= df.groupby(['groupId'])['killPoints'].transform(np.mean)
df['ratioAvgKillPointsGroupAndMatch']= df['avgKillPointsGroup']/ df['avgKillPointsMatch'] 
df['groupRevived']= df.groupby(['groupId'])['revives'].transform(np.sum)
df['groupTeamKills']= df.groupby(['groupId'])['teamKills'].transform(np.sum)
df['avgSpeed']= (df['walkDistance']+ df['swimDistance']+ df['rideDistance'])/ df['matchDuration']
#df= pd.get_dummies(df, columns= ['matchType'], drop_first= True)
le= LabelEncoder()
le.fit(df['matchType'])
df['matchTypeLabels']= le.fit_transform(df['matchType'])
sns.heatmap(df.corr()[(df.corr()>0.75) | (df.corr()< -0.75)])
df.corr()[(df.corr()>0.75) | (df.corr()< -0.75)]
#df.drop(['kills', 'DBNOs', 'killStreaks', 'numGroups', 'rankPoints', 'winPoints', 'damageDealt', 'killPoints', 'killPointsSumMatch', 'killPointsSumGroup', 'avgKillPointsGroup', 'avgKillPointsMatch', 'walkDistance', 'rideDistance', 'swimDistance', 'matchType', 'entryCount', 'groupId', 'matchId'], axis= 1, inplace= True) 
df.drop(['entryCount', 'groupId', 'matchId', 'matchType', 'DBNOs', 'winPoints', 'killPoints', 'killStreaks', 'maxKillPointsGroup', 'revives', 'headshotKills', 'teamKills', 'roadKills', 'vehicleDestroys'], axis= 1, inplace= True)
df.set_index('Id', inplace= True)
df_train= df[df['belongs']== 'train']
df_test= df[df['belongs']== 'test']
df_train.drop('belongs', axis= 1, inplace= True)
df_test.drop('belongs', axis= 1, inplace= True)
df_train.reset_index(inplace= True)
df_train['winPlacePerc']= pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv', usecols= ['winPlacePerc'])
df_train.set_index('Id', inplace= True)
df_train.dropna(inplace= True)
X_train, X_test, y_train, y_test= train_test_split(df_train.drop(['winPlacePerc'], axis= 1), df_train['winPlacePerc'], test_size= 0.3)
model= XGBRegressor(n_estimators= 500, max_depth= 7, n_jobs= -1, min_child_weight= 7, subsample=0.84, colsample_bytree= 0.97, eta=0.3, seed=42)
model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=True, 
    early_stopping_rounds = 10)
plot_features(model, (10,14))
predictions= pd.DataFrame({'winPlacePerc': model.predict(df_test).clip(0,1)}, index= df_test.index)
predictions.to_csv('submission_1.csv')