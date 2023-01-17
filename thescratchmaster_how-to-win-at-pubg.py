import pandas as pd
import matplotlib.pyplot as plt
import gc

import seaborn as sns
import numpy as np

import random
random.seed(42)

from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline

train_f = pd.read_csv("train_V2.csv")
train_f.head(5)
train_f.info()
dtypes = {
        'Id'                : 'object',
        'groupId'           : 'object',
        'matchId'           : 'object',
        'assists'           : 'uint8',
        'boosts'            : 'uint8',
        'damageDealt'       : 'float16',
        'DBNOs'             : 'uint8',
        'headshotKills'     : 'uint8', 
        'heals'             : 'uint8',    
        'killPlace'         : 'uint8',    
        'killPoints'        : 'uint8',    
        'kills'             : 'uint8',    
        'killStreaks'       : 'uint8',    
        'longestKill'       : 'float16',    
        'maxPlace'          : 'uint8',
        'matchType'         : 'object',
        'numGroups'         : 'uint8',    
        'revives'           : 'uint8',    
        'rideDistance'      : 'float16',    
        'roadKills'         : 'uint8',    
        'swimDistance'      : 'float16',    
        'teamKills'         : 'uint8',    
        'vehicleDestroys'   : 'uint8',    
        'walkDistance'      : 'float16',    
        'weaponsAcquired'   : 'uint8',    
        'winPoints'         : 'uint8', 
        'winPlacePerc'      : 'float16' 
}

train = pd.read_csv("train_V2.csv", dtype=dtypes)

train.info()
falseWinPlacePerc = train[train['winPlacePerc'].isna()]['matchId'].values
train = train[-train['matchId'].isin(falseWinPlacePerc)]
droppedColumns = ['Id', 'groupId', 'matchId', 'matchType']
fittedColumns = [col for col in train.columns if col not in droppedColumns]
corr = train[fittedColumns].corr()
plt.figure(figsize=(15,10))
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    linecolor='white',
    linewidths=0.1,
    cmap="BuPu"
)
plt.show()
plt.figure(figsize=(15,10))
corr1=corr.abs()>0.25
sns.heatmap(corr1,annot=True)
plt.show()
killsCategories = train.copy()
killsCategories['killsCategories'] = pd.cut(train['kills'], [-1, 0, 2, 5, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10+kills'])

plt.figure(figsize=(15,10))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=killsCategories)
plt.show()
walkDistanceData = train.copy()
walkDistanceData['walkDistance'] = pd.cut(train['walkDistance'], [-1, 0, 1000, 2000, 10000], labels=['0_steps','1-1000_steps','1000-2000_steps','2000-10000_steps'])

plt.figure(figsize=(15,10))
sns.boxplot(x="walkDistance", y="winPlacePerc", data=walkDistanceData)
plt.show()
train['headshotKills_over_kills'] = train['headshotKills'] / train['kills']
train['headshotKills_over_kills'].fillna(0, inplace=True)
train['headshotKills_over_kills'].replace(np.inf, 0, inplace=True)

train['distancePerKill'] = train['kills'] / (train['walkDistance'] + 1)##No infinity error
train['distancePerKill'].fillna(0, inplace=True)
train['distancePerKill'].replace(np.inf, 0, inplace=True)
#If 100 players join, then a kill matters less than if 2 players joined
train['normalizedKills'] = train['kills'] * ((100 - train['numGroups']) / 100 + 1)
train['normalizedDamage'] = train['damageDealt'] * ((100 - train['numGroups']) / 100 + 1)
train['healsAndBoosts'] = train['heals'] + train['boosts']

train['itemEfficiency'] = train['healsAndBoosts'] / (train['walkDistance'] + 1)##no divide by zeros today
train['itemEfficiency'].fillna(0, inplace=True)
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
corr = train[['headshotKills_over_kills','distancePerKill','normalizedKills','normalizedDamage','healsAndBoosts',
             'itemEfficiency','totalDistance','winPlacePerc']].corr()

plt.figure(figsize=(15,10))
sns.heatmap(
    corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    annot=True,
    linecolor='white',
    linewidths=0.1,
    cmap="RdBu"
)
plt.show()
def group_by_match_TTS(train, test_size=0.2):
    match_ids = train['matchId'].unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids, train_size)

    X = train[train['matchId'].isin(train_match_ids)]
    y = train[-train['matchId'].isin(train_match_ids)]
    
    return X, y
def testFeatures(features):
    results = []
    
    for feature in features:
        gc.collect()
        test_train = pd.read_csv("train_V2.csv", dtype=dtypes)
        invalid_match_ids = test_train[test_train['winPlacePerc'].isna()]['matchId'].values
        test_train = test_train[-test_train['matchId'].isin(invalid_match_ids)]
        
        test_train.drop(columns=['matchType'], inplace=True)
        test_train = feature(test_train)

        cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
        cols_to_fit = [col for col in test_train.columns if col not in cols_to_drop]
        
        train, val = group_by_match_TTS(test_train, 0.2)
    
        model = LinearRegression()
        model.fit(train[cols_to_fit], train['winPlacePerc'])
    
        y_true = val['winPlacePerc']
        y_pred = model.predict(val[cols_to_fit])
        score = mean_absolute_error(y_true, y_pred)
        
        results.append({
            'name': feature.__name__,
            'score': 1 - score
        })
        
    return pd.DataFrame(results, columns=['name', 'score']).sort_values(by='score')
def normal(train):
    return train

def totalDistance(train):
    train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']
    return train

def headshotKills_over_kills(train):
    train['headshotKills_over_kills'] = train['headshotKills'] / train['kills']
    train['headshotKills_over_kills'].fillna(0, inplace=True)
    train['headshotKills_over_kills'].replace(np.inf, 0, inplace=True)
    return train

def normalizedKills(train):
    train['normalizedKills'] = train['kills'] * ((100 - train['numGroups']) / 100 + 1)
    return train

def normalizedDamage(train):   
    train['normalizedDamage'] = train['damageDealt'] * ((100 - train['numGroups']) / 100 + 1)
    return train

def healsAndBoosts(train):
    train['healsAndBoosts'] = train['heals'] + train['boosts']
    return train

def itemEfficiency(train):
    train['healsAndBoosts'] = train['heals'] + train['boosts']
    train['itemEfficiency'] = train['healsAndBoosts'] / (train['walkDistance'] + 1)##no divide by zeros today
    train['itemEfficiency'].fillna(0, inplace=True)
    return train

def distancePerKill(train):
    train['distancePerKill'] = train['kills'] / (train['walkDistance'] + 1)##No infinity error
    train['distancePerKill'].fillna(0, inplace=True)
    train['distancePerKill'].replace(np.inf, 0, inplace=True)
    return train
testFeatures([
    normal,
    totalDistance,
    headshotKills_over_kills,
    normalizedKills,
    normalizedDamage,
    healsAndBoosts,
    itemEfficiency,
    distancePerKill
])
 ##Dropping people who got kills without moving
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
##Dropping people who got kills over 1000m
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
##Dropping people who speedrun like Sonic through the map
train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
##Dropping people who got a suspicion amount of weapons
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
from catboost import CatBoostRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import cross_val_score,KFold
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from  sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
cb = CatBoostRegressor(iterations=500, learning_rate=0.05, 
                       loss_function='MAE',eval_metric='MAE', 
                       depth = 15, use_best_model=True, od_type="Iter", 
                       od_wait=20)
xgb = XGBRegressor(learning_rate=.03,min_child_weight=4,
                 max_depth=10,subsample=.4,
                 n_estimators=500,n_jobs=-1)
GBoost = GradientBoostingRegressor(n_estimators=600, learning_rate=0.05,
                                   max_depth=4, max_features='auto',
                                   min_samples_leaf=15, min_samples_split=5, 
                                   loss='huber', random_state =5)
y = train['winPlacePerc']
train = train.drop(columns=['winPlacePerc','Id','groupId','matchId'],axis=1)
train= pd.get_dummies(train)

x_train,x_test,y_train,y_test=train_test_split(train,y,test_size=0.2)

xgb.fit(x_train,y_train)
cb.fit(x_train,y_train)
GBoost.fit(x_train,y_train)


pred1=xgb.predict(x_test)
pred2=cb.predict(x_test)
pred3=GBoost.predict(x_test)

stacked_predictions=np.column_stack((pred1,pred2,pred3))

meta_model=LinearRegression()
meta_model.fit(stacked_predictions,y_test)
final=meta_model.predict(y_train)

print('r2 score is:', r2_score(y_test,final))
