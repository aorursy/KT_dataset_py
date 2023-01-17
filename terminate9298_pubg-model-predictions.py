import pandas as pd

import numpy as np

import gc

import lightgbm as lgb

import time
train=pd.read_csv('../input/train_V2.csv',nrows=1000000)
train.corr().style.format("{:.2%}").highlight_min()
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    # iterate through all the columns of a dataframe and modify the data type

    #   to reduce memory usage.        

    

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))



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



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df

train = train[train['maxPlace'] > 1]
train.sample(7).T
train['matchType'] = train['matchType'].map({

    'crashfpp':1,

    'crashtpp':2,

    'duo':3,

    'duo-fpp':4,

    'flarefpp':5,

    'flaretpp':6,

    'normal-duo':7,

    'normal-duo-fpp':8,

    'normal-solo':9,

    'normal-solo-fpp':10,

    'normal-squad':11,

    'normal-squad-fpp':12,

    'solo':13,

    'solo-fpp':14,

    'squad':15,

    'squad-fpp':16

    })

train = train[train['winPlacePerc'].notnull()]

train = reduce_mem_usage(train)


matchSizeData = train.groupby(['matchId']).size().reset_index(name='matchSize')

train = pd.merge(train, matchSizeData, on=['matchId'])

del matchSizeData
train = reduce_mem_usage(train)

gc.collect()
train.loc[(train['rankPoints']==-1),'rankPoints'] = 0
train['rankPoints_killPoints'] = train['rankPoints'] + train['killPoints'] 

train['headshotKills_ratio'] = train['headshotKills']/train['kills']

train['killStreaks_ratio'] = train['killStreaks']/train['kills']

train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']

train['killPlace_maxPlace_Ratio']= train['killPlace']/train['maxPlace']

train['distance_weaponAcquired_Ratio'] = train['totalDistance']/train['weaponsAcquired']

train['distance_kills_Ratio'] =train['kills']/train['totalDistance']

train['distance_heals_Ratio'] = train['heals']/train['totalDistance']

train['walkDistance_kills_Ratio'] = train['kills']/train['walkDistance']

train['walkDistance_heals_Ratio'] = train['heals']/train['walkDistance']

train['walk_duration_Ratio'] = train['walkDistance']/train['matchDuration']

train['distance_duration_Ratio'] = train['totalDistance']/train['matchDuration']

train['killplace_kills_Ratio'] = train['kills']/train['killPlace']
train['walkDistancePerc'] = train.groupby('matchId')['walkDistance'].rank(pct=True).values

train['killPerc'] = train.groupby('matchId')['kills'].rank(pct=True).values

train['killPlacePerc'] = train.groupby('matchId')['killPlace'].rank(pct=True).values

train['weaponsAcquiredPerc'] = train.groupby('matchId')['weaponsAcquired'].rank(pct=True).values
train[train == np.Inf] = np.NaN

train[train == np.NINF] = np.NaN

train.fillna(0,inplace = True)
train = reduce_mem_usage(train)
features = list(train.columns)

features.remove('Id')

features.remove('groupId')

features.remove('matchId')

features.remove('matchSize')

features.remove('matchType')

features.remove('winPlacePerc')

print(features)
meanData = train.groupby(['matchId','groupId'])[features].agg('mean')

print(meanData.shape)

display(meanData.head().T)
meanData = meanData.replace([np.inf , np.NINF , np.NaN] , 0)

meanData = reduce_mem_usage(meanData)

meanDataRank = meanData.groupby('matchId')[features].rank(pct=True).reset_index()

meanDataRank.sample(7).T

meanDataRank = reduce_mem_usage(meanDataRank)
train = pd.merge(train, meanData.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])

train = pd.merge(train, meanDataRank, suffixes=["", "_meanRank"], how='left', on=['matchId', 'groupId']) 
del meanData

del meanDataRank

gc.collect()
train = train.join(reduce_mem_usage(train.groupby('matchId')[features].rank(ascending=False).add_suffix('_rankPlace').astype(int)))
train.shape
stdData = train.groupby(['matchId','groupId'])[features].agg('std').replace([np.inf, np.NINF,np.nan], 0)

stdDataRank = reduce_mem_usage(stdData.groupby('matchId')[features].rank(pct=True)).reset_index()

train = pd.merge(train , stdDataRank , suffixes=['','_stdRank'] , how = 'left' , on=['matchId','groupId'])

del stdData

del stdDataRank

gc.collect()
maxData = train.groupby(['matchId','groupId'])[features].agg('std').replace([np.inf, np.NINF,np.nan], 0)

maxDataRank = reduce_mem_usage(maxData.groupby('matchId')[features].rank(pct=True)).reset_index()

train = pd.merge(train , maxDataRank , suffixes=['','_maxRank'] , how = 'left' , on=['matchId','groupId'])

del maxData

del maxDataRank

gc.collect()
fea = train.columns

fea

def split_train_val(data, fraction):

    matchIds = data['matchId'].unique().reshape([-1])

    train_size = int(len(matchIds)*fraction)

    

    random_idx = np.random.RandomState(seed=2).permutation(len(matchIds))

    train_matchIds = matchIds[random_idx[:train_size]]

    val_matchIds = matchIds[random_idx[train_size:]]

    

    data_train = data.loc[data['matchId'].isin(train_matchIds)]

    data_val = data.loc[data['matchId'].isin(val_matchIds)]

    return data_train, data_val
train.dtypes
data_train , data_val = split_train_val(train , .90)

data_train = data_train.drop(columns = ['Id' , 'groupId' , 'matchId'])

data_val = data_val.drop(columns = ['Id' , 'groupId' , 'matchId'])

data_train_y =data_train['winPlacePerc']

data_train = data_train.drop(columns = ['winPlacePerc'])

data_val_y = data_val['winPlacePerc']

data_val = data_val.drop(columns = ['winPlacePerc'])

data_train = np.array(data_train)

data_train_y = np.array(data_train_y)

data_val = np.array(data_val)

data_val_y = np.array(data_val_y)



data_train
train_set = lgb.Dataset( data_train , label = data_train_y)

val_set = lgb.Dataset( data_val , label = data_val_y)


params = {

        "objective" : "regression", 

        "metric" : "mae", 

        "num_leaves" : 60, 

        "learning_rate" : 0.003, 

        "bagging_fraction" : 0.9,

        "bagging_seed" : 0, 

        "num_threads" : 4,

        "colsample_bytree" : 0.5, 

        'lambda_l2':9

}



model = lgb.train(  params, 

                    train_set = train_set,

                    num_boost_round=9400,

                    early_stopping_rounds=200,

                    verbose_eval=100, 

                    valid_sets=[train_set,val_set]

                  )

featureImp = list(model.feature_importance())
len(featureImp)
def test_set_modify(test):

    test['matchType'] = test['matchType'].map({

        'crashfpp':1,

        'crashtpp':2,

        'duo':3,

        'duo-fpp':4,

        'flarefpp':5,

        'flaretpp':6,

        'normal-duo':7,

        'normal-duo-fpp':8,

        'normal-solo':9,

        'normal-solo-fpp':10,

        'normal-squad':11,

        'normal-squad-fpp':12,

        'solo':13,

        'solo-fpp':14,

        'squad':15,

        'squad-fpp':16

        })

    test = reduce_mem_usage(test)



    matchSizeData = test.groupby(['matchId']).size().reset_index(name='matchSize')

    test = pd.merge(test, matchSizeData, on=['matchId'])

    del matchSizeData



    gc.collect()

    test = reduce_mem_usage(test)

    test.loc[(test['rankPoints']==-1),'rankPoints'] = 0



    test['rankPoints_killPoints'] = test['rankPoints'] + test['killPoints'] 

    test['headshotKills_ratio'] = test['headshotKills']/test['kills']

    test['killStreaks_ratio'] = test['killStreaks']/test['kills']

    test['totalDistance'] = test['rideDistance'] + test['walkDistance'] + test['swimDistance']

    test['killPlace_maxPlace_Ratio']= test['killPlace']/test['maxPlace']

    test['distance_weaponAcquired_Ratio'] = test['totalDistance']/test['weaponsAcquired']

    test['distance_kills_Ratio'] =test['kills']/test['totalDistance']

    test['distance_heals_Ratio'] = test['heals']/test['totalDistance']

    test['walkDistance_kills_Ratio'] = test['kills']/test['walkDistance']

    test['walkDistance_heals_Ratio'] = test['heals']/test['walkDistance']

    test['walk_duration_Ratio'] = test['walkDistance']/test['matchDuration']

    test['distance_duration_Ratio'] = test['totalDistance']/test['matchDuration']

    test['killplace_kills_Ratio'] = test['kills']/test['killPlace']



    test['walkDistancePerc'] = test.groupby('matchId')['walkDistance'].rank(pct=True).values

    test['killPerc'] = test.groupby('matchId')['kills'].rank(pct=True).values

    test['killPlacePerc'] = test.groupby('matchId')['killPlace'].rank(pct=True).values

    test['weaponsAcquiredPerc'] = test.groupby('matchId')['weaponsAcquired'].rank(pct=True).values



    test[test == np.Inf] = np.NaN

    test[test == np.NINF] = np.NaN

    test.fillna(0,inplace = True)



    features = list(test.columns)

    features.remove('Id')

    features.remove('groupId')

    features.remove('matchId')

    features.remove('matchSize')

    features.remove('matchType')



    meanData = test.groupby(['matchId','groupId'])[features].agg('mean')

    display(meanData.head().T)



    meanData = meanData.replace([np.inf , np.NINF , np.NaN] , 0)

    meanData = reduce_mem_usage(meanData)



    meanDataRank = meanData.groupby('matchId')[features].rank(pct=True).reset_index()

    meanDataRank = reduce_mem_usage(meanDataRank)



    test = pd.merge(test, meanData.reset_index(), suffixes=["", "_mean"], how='left', on=['matchId', 'groupId'])

    test = pd.merge(test, meanDataRank, suffixes=["", "_meanRank"], how='left', on=['matchId', 'groupId']) 



    del meanData

    del meanDataRank

    gc.collect()



    test = test.join(reduce_mem_usage(test.groupby('matchId')[features].rank(ascending=False).add_suffix('_rankPlace').astype(int)))



    stdData = test.groupby(['matchId','groupId'])[features].agg('std').replace([np.inf, np.NINF,np.nan], 0)

    stdDataRank = reduce_mem_usage(stdData.groupby('matchId')[features].rank(pct=True)).reset_index()

    test = pd.merge(test , stdDataRank , suffixes=['','_stdRank'] , how = 'left' , on=['matchId','groupId'])

    del stdData

    del stdDataRank

    gc.collect()



    maxData = test.groupby(['matchId','groupId'])[features].agg('std').replace([np.inf, np.NINF,np.nan], 0)

    maxDataRank = reduce_mem_usage(maxData.groupby('matchId')[features].rank(pct=True)).reset_index()

    test = pd.merge(test , maxDataRank , suffixes=['','_maxRank'] , how = 'left' , on=['matchId','groupId'])

    del maxData

    del maxDataRank

    gc.collect()

    

    

    return test





test = pd.read_csv('../input/test_V2.csv')
test = test_set_modify(test)
features_test = test.columns
test = test.drop(columns=['Id','groupId' , 'matchId'])

test = np.array(test)

print('Predicting Start : ', time.ctime())

y_pred=model.predict(test, num_iteration=model.best_iteration)

print('Prediction end : ', time.ctime())
df_sub = pd.DataFrame()

df_test = pd.read_csv('../input/test_V2.csv')

df_test = reduce_mem_usage(df_test)

df_sub['Id'] = df_test['Id']

df_sub['winPlacePerc'] = y_pred
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()

df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()

# we Find the max_rank for a group in match

df_sub_group = df_sub_group.merge(df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(),on="matchId", how="left")



df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")

# so we find the rank of each player in match 

# So this is whole process is to find that we find the rank of player in match as

# The predications can be conflicting and this process will remove rank  anomally



df_sub["winPlacePerc"] = df_sub["adjusted_perc"]
df_sub.sample(10).T
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0

df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

subset = df_sub.loc[df_sub.maxPlace > 1]

gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap

df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case

df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0

assert df_sub["winPlacePerc"].isnull().sum() == 0



df_sub[["Id", "winPlacePerc"]].to_csv("submission_2.csv", index=False)

print(df_sub['winPlacePerc'].describe())