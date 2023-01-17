# debug mode
debug = False;
debug_rows = 10000;

# import
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb

from xgboost import XGBRegressor
from sklearn import model_selection
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

import gc, sys
gc.enable()

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns', 100)

%matplotlib inline
# if(debug):
#     plt.style.use("dark_background")

if(debug):
    train = pd.read_csv('../input/train_V2.csv', nrows = debug_rows)
    test  = pd.read_csv('../input/test_V2.csv')
else:
    train = pd.read_csv('../input/train_V2.csv', nrows = debug_rows)
    test  = pd.read_csv('../input/test_V2.csv')
train.shape
test.shape
train.isnull().sum()
train.dropna(axis=0, how='all')
print(train.isnull().any().any())
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

# train = reduce_mem_usage(train)
# test = reduce_mem_usage(test)
train.describe()
train.quantile(q=[0.10, 0.90], numeric_only=True)

assists = train['assists']
assists.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=assists, y=train['winPlacePerc'], ax=ax)
fig, ax = plt.subplots(figsize=(16,8))
sns.distplot(assists, kde=False)
assists[assists>10].count()
boosts = train['boosts']
boosts.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=boosts, y=train['winPlacePerc'], ax=ax)
fig, ax = plt.subplots(figsize=(16,8))
sns.distplot(boosts, kde=False)
train[['boosts', 'heals']].corr()
eda = train['damageDealt']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'], fit_reg=False, ax=ax)
eda[eda>15].count()
train[(train['damageDealt']>400) & (train['winPlacePerc']==0.0) & (train['matchType']=='solo')].head()

eda = train['DBNOs']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'], ax=ax)
eda[eda>10].count()

eda = train['walkDistance']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'], fit_reg=False,ax=ax)
cheater = train[(train['walkDistance']<=50.0)&(train['damageDealt']>0.0)]
cheater.head()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=cheater['walkDistance'], y=cheater['winPlacePerc'], fit_reg=False,ax=ax)
train[train['walkDistance']>10000]
eda = train['headshotKills']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'],ax=ax)
train[train['headshotKills']>10]
eda = train['heals']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'],ax=ax)
train[train['heals']>40].describe()
eda = train['killPlace']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'],ax=ax)

eda = train['killPoints']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'],fit_reg=False, ax=ax)
eda = train['kills']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'], ax=ax)
sns.heatmap(train[['damageDealt','kills', 'headshotKills', 'DBNOs']].corr(), annot=True)
train[train['kills']>30].describe()
sns.regplot(x=train[train['kills']>10]['kills'], y=train[train['kills']>10]['walkDistance'], fit_reg=False)
eda = train['killStreaks']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'], ax=ax)
sns.heatmap(train[['damageDealt','kills', 'headshotKills', 'DBNOs', 'killStreaks']].corr(), annot=True)
train[train['killStreaks']>10].head()
eda = train['longestKill']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'],fit_reg=False, ax=ax)
train[(train['longestKill']<0.01) & (train['kills']!=0)].describe()
sns.regplot(x=train[(train['longestKill']<1.0) & (train['kills']!=0)]['longestKill'], y=train[(train['longestKill']<1.0) & (train['kills']!=0)]['winPlacePerc'], fit_reg=False)
eda = train['matchDuration']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'],fit_reg=False, ax=ax)
eda = train['rankPoints']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'],fit_reg=False, ax=ax)
train[train['rankPoints']==-1].describe()
eda = train['revives']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'], ax=ax)
eda = train['roadKills']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'],ax=ax)
train[(train['rideDistance']==0)&(train['roadKills']>0)]
eda = train['swimDistance']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'],fit_reg=False, ax=ax)
eda = train['teamKills']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'], ax=ax)
train[train['teamKills']>3]
eda = train['vehicleDestroys']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'], ax=ax)
eda = train['walkDistance']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'],fit_reg=False, ax=ax)
eda = train['weaponsAcquired']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.boxplot(x=eda, y=train['winPlacePerc'], ax=ax)
train[(train['weaponsAcquired']>30) & (train['walkDistance']<100)].describe()
eda = train['winPoints']
eda.describe()
fig, ax = plt.subplots(figsize=(16,8))
sns.regplot(x=eda, y=train['winPlacePerc'],fit_reg=False, ax=ax)
df = train # just to save train_df safe

df = df.drop(df[(df['walkDistance']<10.0) & (df['damageDealt']>0)].index)
df = df.drop(df[(df['walkDistance']<10.0) & (df['kills']>10)].index)
df = df.drop(df[(df['walkDistance']<100.0) & (df['weaponsAcquired']>30)].index)
df = df.drop(df[(df['walkDistance']<10.0) & (df['heals']>100)].index)
df = df.drop(df[(df['walkDistance']<10.0) & (df['headshotKills']>5)].index)
df = df.drop(df[(df['walkDistance']<10.0) & (df['headshotKills']>5)].index)

# unrated guys (killPoints)
df['unrated_kill'] = 0
df.loc[df['killPoints']==0, 'unrated_kill']=1

# poor on kill points
df['poor_kills'] = 0
df.loc[(df['killPoints']>10) & (df['killPoints']<800), 'poor_kills'] = 1
df.loc[(df['killPoints']>10) & (df['killPoints']<800), 'killPoints'] = 0

# killPerDamage
df['killPerDamage'] = df['kills']/df['damageDealt']
df = df.fillna(0)

# drop savage killer (kill streak > 10)
df = df.drop(df[df['killStreaks']>=10].index)


# rank unrated players
df['unrated_rank'] = 0
df.loc[df['rankPoints']==-1, 'unrated_rank']=1

# roadDistance glitch drop
df = df.drop(df[(df['rideDistance']==0.0) & (df['roadKills']>0)].index)

# insane weapon scavenger = cheater. drop
df = df.drop(df[(df['weaponsAcquired']>30) & (df['walkDistance']<100)].index)

# winpoints unrated
df['unrated_win'] = 0
df.loc[(df['winPoints']==-1) | (df['winPoints'] == 0), 'unrated_win']=1

# poor on winpoints
df['poor_wins'] = 0
df.loc[(df['winPoints']>250) & (df['winPoints']<1200), 'poor_wins'] = 1
df.loc[(df['winPoints']>250) & (df['winPoints']<1200), 'killPoints'] = 0

print('removed:' + str(train['Id'].count() - df['Id'].count()))
df.head()


# thanks to awesome https://www.kaggle.com/chocozzz/lightgbm-baseline

def feature_engineering(is_train=True,debug=True):
    test_idx = None
    if is_train: 
        print("processing train.csv")
        if debug == True:
            df = pd.read_csv('../input/train_V2.csv', nrows=1000000)
        else:
            df = pd.read_csv('../input/train_V2.csv')

        df = df[df['maxPlace'] > 1]
    else:
        print("processing test.csv")
        if debug == True:
            df = pd.read_csv('../input/test_V2.csv')
        else:
            df = pd.read_csv('../input/test_V2.csv')
        test_idx = df.Id
    
    # df = reduce_mem_usage(df)
    #df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    
    # df = df[:100]
    
    print("remove some columns")
    target = 'winPlacePerc'
    
    if(is_train):
        print("removing cheaters")
        df = df.drop(df[(df['walkDistance']<10.0) & (df['damageDealt']>0)].index)
        df = df.drop(df[(df['walkDistance']<10.0) & (df['kills']>10)].index)
        df = df.drop(df[(df['walkDistance']<100.0) & (df['weaponsAcquired']>30)].index)
        df = df.drop(df[(df['walkDistance']<10.0) & (df['heals']>100)].index)
        df = df.drop(df[(df['walkDistance']<10.0) & (df['headshotKills']>5)].index)
        df = df.drop(df[(df['walkDistance']<10.0) & (df['headshotKills']>5)].index)

        # drop savage killer (kill streak > 10)
        df = df.drop(df[df['killStreaks']>=10].index)

        # roadDistance glitch drop
        df = df.drop(df[(df['rideDistance']==0.0) & (df['roadKills']>0)].index)

        # insane weapon scavenger = cheater. drop
        df = df.drop(df[(df['weaponsAcquired']>30) & (df['walkDistance']<100)].index)

    # unrated guys (killPoints)
    df['unrated_kill'] = 0
    df.loc[df['killPoints']==0, 'unrated_kill']=1

    # poor on kill points
    df['poor_kills'] = 0
    df.loc[(df['killPoints']>10) & (df['killPoints']<800), 'poor_kills'] = 1
    df.loc[(df['killPoints']>10) & (df['killPoints']<800), 'killPoints'] = 0

    


    # rank unrated players
    df['unrated_rank'] = 0
    df.loc[df['rankPoints']==-1, 'unrated_rank']=1

    

    # winpoints unrated
    df['unrated_win'] = 0
    df.loc[(df['winPoints']==-1) | (df['winPoints'] == 0), 'unrated_win']=1

    # poor on winpoints
    df['poor_wins'] = 0
    df.loc[(df['winPoints']>250) & (df['winPoints']<1200), 'poor_wins'] = 1
    df.loc[(df['winPoints']>250) & (df['winPoints']<1200), 'killPoints'] = 0

   
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    
    print("Removing Na's From DF")
    df.fillna(0, inplace=True)

    
    features = list(df.columns)
    features.remove("Id")
    features.remove("matchId")
    features.remove("groupId")
    features.remove("matchType")
    
    # matchType = pd.get_dummies(df['matchType'])
    # df = df.join(matchType)    
    
    y = None
    
    
    if is_train: 
        print("get target")
        y = np.array(df.groupby(['matchId','groupId'])[target].agg('mean'), dtype=np.float64)
        features.remove(target)

    print("get group mean feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('mean')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    
    if is_train: df_out = agg.reset_index()[['matchId','groupId']]
    else: df_out = df[['matchId','groupId']]

    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])
    
    # print("get group sum feature")
    # agg = df.groupby(['matchId','groupId'])[features].agg('sum')
    # agg_rank = agg.groupby('matchId')[features].agg('sum')
    # df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    # df_out = df_out.merge(agg_rank.reset_index(), suffixes=["_sum", "_sum_pct"], how='left', on=['matchId', 'groupId'])
    
    print("get group max feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('max')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group min feature")
    agg = df.groupby(['matchId','groupId'])[features].agg('min')
    agg_rank = agg.groupby('matchId')[features].rank(pct=True).reset_index()
    df_out = df_out.merge(agg.reset_index(), suffixes=["", ""], how='left', on=['matchId', 'groupId'])
    df_out = df_out.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])
    
    print("get group size feature")
    agg = df.groupby(['matchId','groupId']).size().reset_index(name='group_size')
    df_out = df_out.merge(agg, how='left', on=['matchId', 'groupId'])
    
    print("get match mean feature")
    agg = df.groupby(['matchId'])[features].agg('mean').reset_index()
    df_out = df_out.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])
    
    # print("get match type feature")
    # agg = df.groupby(['matchId'])[matchType.columns].agg('mean').reset_index()
    # df_out = df_out.merge(agg, suffixes=["", "_match_type"], how='left', on=['matchId'])
    
    print("get match size feature")
    agg = df.groupby(['matchId']).size().reset_index(name='match_size')
    df_out = df_out.merge(agg, how='left', on=['matchId'])
    
    print("Adding Features")
 
    df['headshotrate'] = df['kills']/df['headshotKills']
    df['killStreakrate'] = df['killStreaks']/df['kills']
    df['healthitems'] = df['heals'] + df['boosts']
    df['totalDistance'] = df['rideDistance'] + df["walkDistance"] + df["swimDistance"]
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['distance_over_weapons'] = df['totalDistance'] / df['weaponsAcquired']
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['killsPerWalkDistance'] = df['kills'] / df['walkDistance']
    df["skill"] = df["headshotKills"] + df["roadKills"]
    
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    print("Removing Na's From DF")
    df.fillna(0, inplace=True)
    
    df_out.drop(["matchId", "groupId"], axis=1, inplace=True)

    X = df_out
    
    feature_names = list(df_out.columns)

    del df, df_out, agg, agg_rank
    gc.collect()

    return X, y, feature_names, test_idx
x_train, y_train, train_columns, _ = feature_engineering(True, debug=debug)
x_test, _, _ , test_idx = feature_engineering(False, debug=debug)

sns.heatmap(x_train.head(1000).corr())

# Thanks and credited to https://www.kaggle.com/gemartin who created this wonderful mem reducer
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df

x_train = reduce_mem_usage(x_train)
x_test = reduce_mem_usage(x_test)
#excluded_features = []
#use_cols = [col for col in df_train.columns if col not in excluded_features]
gc.collect();
train_index = round(int(x_train.shape[0]*0.8))
dev_X = x_train[:train_index] 
val_X = x_train[train_index:]
dev_y = y_train[:train_index] 
val_y = y_train[train_index:] 
gc.collect();

# custom function to run light gbm model
def run_lgb(train_X, train_y, val_X, val_y, x_test):
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 'early_stopping_rounds':200,
              "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.7,
               "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7
             }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, valid_sets=[lgtrain, lgval], early_stopping_rounds=200, verbose_eval=1000)
    
    pred_test_y = model.predict(x_test, num_iteration=model.best_iteration)
    return pred_test_y, model

# Training the model #
pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, x_test)
pred_test


print(pred_test.shape[0])
pred_test

df_sub = pd.read_csv("../input/sample_submission_V2.csv")
df_sub['winPlacePerc'] = pred_test
df_sub.head()

if(debug):
    df_sub = pd.read_csv("../input/sample_submission_V2.csv", nrows=pred_test.shape[0])
    df_test = pd.read_csv("../input/test_V2.csv", nrows=pred_test.shape[0])
else:
    df_sub = pd.read_csv("../input/sample_submission_V2.csv")
    df_test = pd.read_csv("../input/test_V2.csv")
df_sub['winPlacePerc'] = pred_test
# Restore some columns
df_sub = df_sub.merge(df_test[["Id", "matchId", "groupId", "maxPlace", "numGroups"]], on="Id", how="left")

# Sort, rank, and assign adjusted ratio
df_sub_group = df_sub.groupby(["matchId", "groupId"]).first().reset_index()
df_sub_group["rank"] = df_sub_group.groupby(["matchId"])["winPlacePerc"].rank()
df_sub_group = df_sub_group.merge(
    df_sub_group.groupby("matchId")["rank"].max().to_frame("max_rank").reset_index(), 
    on="matchId", how="left")
df_sub_group["adjusted_perc"] = (df_sub_group["rank"] - 1) / (df_sub_group["numGroups"] - 1)

df_sub = df_sub.merge(df_sub_group[["adjusted_perc", "matchId", "groupId"]], on=["matchId", "groupId"], how="left")
df_sub["winPlacePerc"] = df_sub["adjusted_perc"]

# Deal with edge cases
df_sub.loc[df_sub.maxPlace == 0, "winPlacePerc"] = 0
df_sub.loc[df_sub.maxPlace == 1, "winPlacePerc"] = 1

# Align with maxPlace
# Credit: https://www.kaggle.com/anycode/simple-nn-baseline-4
subset = df_sub.loc[df_sub.maxPlace > 1]
gap = 1.0 / (subset.maxPlace.values - 1)
new_perc = np.around(subset.winPlacePerc.values / gap) * gap
df_sub.loc[df_sub.maxPlace > 1, "winPlacePerc"] = new_perc

# Edge case
df_sub.loc[(df_sub.maxPlace > 1) & (df_sub.numGroups == 1), "winPlacePerc"] = 0
assert df_sub["winPlacePerc"].isnull().sum() == 0

df_sub[["Id", "winPlacePerc"]].to_csv("submission_adjusted.csv", index=False)
df_sub
#small test in small batch data
# train_small = df.sample(10000)
# feature_list = ['DBNOs','headshotKills','heals','longestKill','assists','walkDistance', 'boosts','damageDealt', 'damageDealer','healer','deadEye','walker','booster','winPlacePerc']

# train_small=train_small.drop('Id', axis=1)
# train_small=train_small.drop('groupId', axis=1)
# train_small=train_small.drop('matchId', axis=1)
# train_small=train_small.drop('matchType', axis=1)

# train_small_batch = train_small.copy()
# corr = train_small_batch.corr()
# fig, ax = plt.subplots(figsize=(20,20))
# sns.heatmap(corr, annot=True,ax = ax)
# train_df, test_df = model_selection.train_test_split(train_small_batch, test_size=0.3, random_state=49)
# train_df_y = train_df[['winPlacePerc']]
# train_df_x = train_df.copy().drop('winPlacePerc', axis=1)
# test_df_y = test_df[['winPlacePerc']]
# test_df_x = test_df.copy().drop('winPlacePerc', axis=1)

# clf = XGBRegressor()
# clf_cv = model_selection.GridSearchCV(clf, {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}, verbose=1)
# clf_cv.fit(train_df_x, train_df_y)
# print(clf_cv.best_params_, clf_cv.best_score_)
# clf = XGBRegressor(max_depth=4, n_estimators=200)
# clf.fit(train_df_x, train_df_y)
# pred = clf.predict(test_df_x)
# rmse = np.sqrt(mean_absolute_error(test_df_y, pred))
# mean_pred = [train_df_y.mean() for i in range(len(test_df_y))]
# rmse_base = np.sqrt(mean_absolute_error(test_df_y, mean_pred))

# print('trained feature list: ' + str(feature_list))

# print(rmse_base)
# print(rmse)

# xgb.plot_importance(clf, max_num_features=100)

# _test  = pd.read_csv('../input/test_V2.csv')
# test = _test.copy()
# # test['damageDealer']=0
# # test.loc[test['damageDealt']>186.70, 'damageDealer'] = 1

# # test['deadEye'] = 0
# # test.loc[test['headshotKills']>3, 'deadEye'] = 1

# # test['healer'] = 0
# # test.loc[test['heals']>=5, 'healer'] = 1

# # test['walker'] = 0
# # test.loc[test['walkDistance']>3000.0, 'walker'] = 1

# # test['booster'] = 0
# # test.loc[test['boosts']>=4.0, 'booster'] = 1

# # test['sniper'] = 0
# # test.loc[test['headshotKills']>=2.0, 'sniper'] = 1

# test['killPerddamage'] = test['kills']/test['damageDealt']

# test=test.drop('Id', axis=1)
# test=test.drop('groupId', axis=1)
# test=test.drop('matchId', axis=1)
# test=test.drop('matchType', axis=1)


# pred = clf.predict(test)

# submission = pd.DataFrame({'Id':_test['Id'], 'winPlacePerc':pred})
# submission.head()