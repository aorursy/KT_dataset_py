import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from sklearn.metrics import mean_absolute_error, r2_score

import xgboost
from lightgbm import LGBMRegressor
pd.set_option('display.max_columns', 500)


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import warnings
warnings.filterwarnings("ignore")
folder = "../input/"
# folder = "./input/"
print(os.listdir(folder))

NORMFACTOR = 1000
TRAIN_SIZE = 0.9
RANDOM_STATE = 212
EARLY_STOP_ROUNDS = 10
DEBUG = False

# Any results you write to the current directory are saved as output.
cID = ['Id', 'groupId', 'matchId']
cOrigGP= ['assists', 'boosts', 'damageDealt',
       'DBNOs', 'headshotKills', 'heals', 
       'kills', 'killStreaks', 'longestKill', 'revives',
       'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
       'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'killPlace'] # Gameplay information
cRKAll = ['matchDuration', 'killPoints', 'maxPlace', 'numGroups','rankPoints','winPoints'] # Match derived information
cRKPop =['matchDuration', 'maxPlace', 'numGroups']
cMT = ['matchType'] # the only categorical variable
cY = ['winPlacePerc']

cAgg = ['totalKills','netKills', 'totalDistance','teamwork','itemUse','netKillsPerDist','netItemsPerDist','headsPerKill','netWeaponsPerDist']
cGPAggMOnly = ['teamSize']
cGP = cOrigGP+cAgg
cGPAggM = cGP+cGPAggMOnly

cGPNZip = [(col, col+"N") for col in cGP]
cGPN = [col+"N" for col in cGP] # Normalized for match then team
cGPT = [col+"T" for col in cGP]
cGPM = [col+"M" for col in cGPAggM]

cGPMInherit = [col+"M" for col in cGP]

cGPAll = cGPAggM+cGPT+cGPM+cGPN

def preprocess(train, cOrigFeat):
    train = train.copy()
    cOrig = train.columns.values
    # Teamkills not included in kills, roadKills is included
    train['teamSizeFeat'] = train.groupby('groupId')['groupId'].transform('count')
    train['totalKills'] = train['DBNOs'] + train['kills'] + train['teamKills']
    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['walkDistance']
    train['netKills'] = train['DBNOs'] + train['kills'] - train['teamKills']
    train['teamwork'] = train['assists'] - train['teamKills']
    train['itemUse'] = train['boosts'] + train['heals']
    train['netKillsPerDist'] = train['netKills']/(train['totalDistance']+1)
    train['netItemsPerDist'] = train['itemUse']/(train['totalDistance']+1)
    train['headsPerKill'] = train['headshotKills']/(train['totalKills']+1)
    train['netWeaponsPerDist'] = train['weaponsAcquired']/(train['totalDistance']+1)
    train['totalMatchPlayers'] = train.groupby('matchId')['Id'].transform('count')
    train['killsInvolved'] = train['DBNOs'] + train['kills'] + train['teamKills'] + train['assists']
    train['pctMatchKills'] = train['totalKills']/train['totalMatchPlayers']
    train['pctMatchKillInvolvement'] = train['killsInvolved']/train['totalMatchPlayers']
    cEng = [col for col in train.columns.values if col not in cOrig]
    cBase = list(cOrigFeat) + list(cEng)
    
    # Below are utility features which should not be trained on or modified
    train['teamSize'] = train['teamSizeFeat']
    train['sampleWeight'] = 1/train['teamSize']
    cNotFeatures = [col for col in train.columns.values if col not in cBase]
    
    print('features done, scaling by match')
    train[cBase] = minMidMaxGroupScale(train.groupby('matchId'), 0.75, base=train, col=cBase)
    
    print('match aggregation done, aggregating teams')
    # The only thing that matters is teamwork. The features will be replace by sum by team, and min/max/mean by team will be added as well
    train = train.merge(train.groupby('groupId')[cBase].transform(np.min), how='left', suffixes=["","_min"], left_index=True, right_index=True)
    train = train.merge(train.groupby('groupId')[cBase].transform(np.max), how='left', suffixes=["","_max"], left_index=True, right_index=True)
    train = train.merge(train.groupby('groupId')[cBase].transform(np.mean), how='left', suffixes=["","_mean"], left_index=True, right_index=True)
    train[cBase] = train.groupby('groupId')[cBase].transform(np.sum)
    
    cDerived = [col for col in train.columns.values if col not in cBase+cNotFeatures]
    cAll = cBase+cDerived
    return (train, cAll)
    
def minMidMaxGroupScale(g, midQuant, base, col=cGPAggM):
    gMax = g[col].transform(np.max)
    print("max done, quantiles")
    g75 = g[col].transform(lambda x: np.quantile(x, midQuant))
    print("quantiles done, mins")
    gMin = g[col].transform(np.min)
#     outdf = base[col]
    print("mins done, time for big calc")
    outdf = (base[col] <= g75[col])*(base[col]-gMin[col])/(g75[col]-gMin[col]).replace(to_replace=0,value=1)*midQuant+(base[col] > g75[col])*((base[col]-g75[col])/(gMax[col]-g75[col]).replace(to_replace=0,value=1)*(1-midQuant) + midQuant) 
    print("big calc done")
    return outdf   
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
def shape(df):
    return '{:,} rows - {:,} columns'.format(df.shape[0], df.shape[1])

def train_validation(df, train_size=TRAIN_SIZE):
    
    unique_games = df.matchId.unique()
    train_index = round(int(unique_games.shape[0]*train_size))
    
    np.random.shuffle(unique_games)
    
    train_id = unique_games[:train_index]
    validation_id = unique_games[train_index:]
    
    train = df[df.matchId.isin(train_id)]
    validation = df[df.matchId.isin(validation_id)]
    
    return train, validation
df = pd.read_csv(folder+'train_V2.csv')
if DEBUG:
    df = df.head(100000)
time_0  = datetime.datetime.now()

df = reduce_mem_usage(df)
df, cAll = preprocess(df, cOrigGP)
print('Feature engineering done')
df = reduce_mem_usage(df)

time_1  = datetime.datetime.now()
print('Preprocessing took {} seconds'.format((time_1 - time_0).seconds))

df.head()
# lgbm code stolen from https://www.kaggle.com/mm5631/ml-workflow-data-science-approach
train, validation = train_validation(df)
train.head()
train_weights = (1/train['teamSize'])
validation_weights = (1/validation['teamSize'])
X_train = train[cAll]
X_test = validation[cAll]

y_train = train[cY]
y_test = validation[cY]

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
    ax.set_ylim(np.nanmax(losses), 0)
    ax.set_xlim(0,100)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Mean Asbolute Error')
    ax.set_title('Evolution of MAE over training iterations')
    ax.plot(losses, color='grey');
    
plot_training(lgbm)
test = reduce_mem_usage(pd.read_csv(folder+'test_V2.csv'))
if DEBUG:
    test = test.head(10000)
time_0  = datetime.datetime.now()
test, cAll = preprocess(test, cOrigGP)
time_1  = datetime.datetime.now()
print('Preprocessing took {} seconds'.format((time_1 - time_0).seconds))
test = reduce_mem_usage(test)
test.head()
X_sub = test[cAll]
shape(X_sub)
y_sub_unfull = lgbm.predict(X_sub, num_iteration=lgbm.best_iteration_)
sub = test[['Id']]
sub['winPlacePerc'] = y_sub_unfull
sub.head()

sub.to_csv('sub_LGBM_unfull.csv',index=False)
X_train = df[cAll]
y_train = df[cY]
train_weights = (1/df['teamSize'])
time_0 = datetime.datetime.now()

lgbm = LGBMRegressor(objective='mae', n_estimators=250,  
                     learning_rate=0.3, num_leaves=200, 
                     n_jobs=-1,  random_state=RANDOM_STATE, verbose=1)

lgbm.fit(X_train, y_train, sample_weight=train_weights)

time_1  = datetime.datetime.now()

print('Training took {} seconds. Best iteration is {}'.format((time_1 - time_0).seconds, lgbm.best_iteration_))
y_sub_full = lgbm.predict(X_sub, num_iteration=lgbm.best_iteration_)
sub2 = test[['Id']]
sub2['winPlacePerc'] = y_sub_full
sub2.head()
sub2.to_csv('sub_LGBM_full.csv',index=False)