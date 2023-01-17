# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# from tqdm import tqdm_notebook

import gc



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

np.random.seed(1)



small_subset = True

kernel = True

filesdir = 'data/'

if kernel:

    os.listdir("../input")

    filesdir = '../input/'



# Any results you write to the current directory are saved as output.
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



# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

def reduce_mem_usage(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.

    """

    start_mem = df.memory_usage().sum() / 1024**2

    

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

                #if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                #    df[col] = df[col].astype(np.float16)

                #el

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        #else:

            #df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(

        start_mem, end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
train = pd.read_csv(filesdir + 'train_V2.csv', dtype=dtypes)

test = pd.read_csv(filesdir + 'test_V2.csv', dtype=dtypes)
train.info()
def reload(subset=1):

    train = pd.read_csv(filesdir + 'train_V2.csv', dtype=dtypes)

    test = pd.read_csv(filesdir + 'test_V2.csv', dtype=dtypes)

    gc.collect()

    train = train.dropna()

    return train, test
# plot whole DataFrame

def hist_df(df):

    df.hist(figsize=(15,30), layout=(10, 3))

    plt.show()

    

def plot_players_joined(df):

    data = df[df['playersJoined']>49]

    plt.figure(figsize=(15,10))

    sns.countplot(data['playersJoined'])

    plt.title("Players Joined",fontsize=15)

    plt.show()
# drop nan rows

train = train.dropna()



# sample amount of matches to get smaller dataFrame that is suitable for feature selection

def get_subset_from_indexes(df, amount=500):

    uniques = pd.Series.unique(df['matchId'])

    assert amount <= uniques.shape[0], "amount of matches should be less than total amount of matches"

    matchIds = np.random.choice(uniques, size=amount, replace=False)

    return df[df['matchId'].isin(matchIds)]



# remove outliers by clipping values by 1% - 99% value (bounding)

def clip(df):

    cols_to_drop = ['Id', 'groupId', 'matchId','matchType', 'winPlacePerc']

    features_to_fit = [col for col in df.columns if col not in cols_to_drop]

    df[features_to_fit] = df[features_to_fit].clip(df[features_to_fit].quantile(0.01), df[features_to_fit].quantile(0.99), axis=1)    

    return df



def normalize_df(df):

    df = df.dropna()

    df = clip(df)



train, test = reload()

    

    
df = clip(train)
df.head()
def oneHotEncodeGameType(df, colName='matchType', prefix='match_'):

    oneHotEncoded = pd.get_dummies(df[colName],prefix=prefix, drop_first=True)

    returnDf = pd.concat([df, oneHotEncoded], axis=1)

    return returnDf



def add_walk_distance_sqrt(df):

    df['walkDistance_sqrt'] = df['walkDistance'].apply(

        lambda x: np.sqrt(x) 

    )

    return df



def add_walk_distance_log(df):

    df['walkDistance_log'] = df['walkDistance'].apply(

        lambda x: np.log(x + 1) 

    )

    return df



def add_damage_dealt_log(df):

    df['damageDealt_log'] = df['damageDealt'].apply(

        lambda x: np.log(x + 1)

    )

    return df



def players_in_team(df):

    agg = df.groupby(['groupId']).size().to_frame('players_in_team')

    return df.merge(agg, how='left', on=['groupId'])



def total_distance(df):

    df['total_distance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']

    return df



def items(df):

    df['items'] = df['heals'] + df['boosts']

    return df



def headshotKills_over_kills(df):

    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']

    df['headshotKills_over_kills'].fillna(0, inplace=True)

    df['headshotKills_over_kills'].replace(np.inf, 0, inplace=True)

    return df



def killPlace_over_maxPlace(df):

    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']

    df['killPlace_over_maxPlace'].fillna(0, inplace=True)

    df['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)

    return df



def walkDistance_over_heals(df):

    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']

    df['walkDistance_over_heals'].fillna(0, inplace=True)

    df['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)

    return df



def walkDistance_over_kills(df):

    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']

    df['walkDistance_over_kills'].fillna(0, inplace=True)

    df['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)

    return df



def teamwork(df):

    df['teamwork'] = df['assists'] + df['revives']

    return df



def add_players_joined(df):

    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')

    return df



def add_kills_norm(df):

    df['killsPercentage'] = df['kills']*(df['playersJoined']/100)

    return df



def add_damagedone_norm(df):

    try: 

        df['damageDonePercentage'] = df['damageDealt'] / (100 * df['playersJoined'])

    except:

        pass

    return df



def min_by_team(df, agg_cols):

    agg = df.groupby(['matchId','groupId'])[agg_cols].min()

    returndf = df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

    return returndf



def max_by_team(df, agg_cols):

    agg = df.groupby(['matchId', 'groupId'])[agg_cols].max()

    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])



def sum_by_team(df, agg_cols):

    agg = df.groupby(['matchId', 'groupId'])[agg_cols].sum()

    return df.merge(agg, suffixes=['', '_sum'], how='left', on=['matchId', 'groupId'])



def median_by_team(df, agg_cols):

    agg = df.groupby(['matchId', 'groupId'])[agg_cols].median()

    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])



def mean_by_team(df, agg_cols):

    agg = df.groupby(['matchId', 'groupId'])[agg_cols].mean()

    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])



def rank_by_team(df, agg_cols):

    agg = df.groupby(['matchId', 'groupId'])[agg_cols].mean()

    print('aggregation cols', agg_cols)

#     print(agg.columns)

    agg = agg.groupby('matchId')[agg_cols].rank(pct=True)

    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])



def get_X_Y_fromdf(df):

    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']

    fitcol = ['winPlacePerc']

    features_to_fit = [col for col in df.columns if col not in cols_to_drop]

    X = df[features_to_fit]

    y = None

    if fitcol[0] in df.columns:

        y = df[fitcol]

    return X, y



fe_functions = [

    clip,

    add_players_joined,

    add_kills_norm,

    add_damagedone_norm,

#     oneHotEncodeGameType,

    add_walk_distance_sqrt,

    add_walk_distance_log,

    add_damage_dealt_log,

    players_in_team,

    total_distance,

    items,

    headshotKills_over_kills,

    killPlace_over_maxPlace,

    walkDistance_over_heals,

    walkDistance_over_kills,

    teamwork

]



fe_agg_functions = [

    rank_by_team

#     ,

#     min_by_team,

#     max_by_team,

#     sum_by_team,

#     median_by_team,

#     mean_by_team

]
df_subset = get_subset_from_indexes(train).drop(columns='matchType')

pd.options.mode.chained_assignment = None  # disable the copy warning assignment

# df_subset.info()

def add_basic_fe(df):

#     cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']

#     features_to_fit = [col for col in df.columns if col not in cols_to_drop]

    start_c = len(df.columns)

    for func in fe_functions:

        start = len(df.columns)

        df = func(df)

        end = len(df.columns)

        print(f"Added {end - start} feature(s) with {func.__name__} function")

        gc.collect()

    end_c = len(df.columns)

    print(f'in total added {end_c - start_c} columns.')

    return df



def add_agg_fe(df):

    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']

    features_to_fit = [col for col in df.columns if col not in cols_to_drop]

    start_c = len(df.columns)

    print('start', df.columns)

    for func in fe_agg_functions:

        start = len(df.columns)

        df = func(df, features_to_fit)

        end = len(df.columns)

        print(f"Added {end - start} feature(s) with {func.__name__} function")

        gc.collect()

    end_c = len(df.columns)

    print(f'in total added {end_c - start_c} columns.')

    return df



def add_df_features(df):

    df_subset_with_basic = add_basic_fe(df)    

    df_subset_with_basic_agg = add_agg_fe(df_subset_with_basic)

    return df_subset_with_basic_agg



df_subset_with_basic_agg = add_df_features(df_subset)

df_subset_with_basic_agg.info()
# def cross_val_score_MSE(model):
from sklearn.feature_selection import SelectFromModel

from lightgbm import LGBMRegressor



cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']

fitcol = ['winPlacePerc']

features_to_fit = [col for col in df_subset_with_basic_agg.columns if col not in cols_to_drop]

# define X and y

X_train = df_subset_with_basic_agg[features_to_fit]

y_train = df_subset_with_basic_agg[fitcol]



lgbr=LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,

            reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)



embeded_lgb_selector = SelectFromModel(lgbr, threshold='1.25*median')

embeded_lgb_selector.fit(X_train, y_train)
print('Start cols len: ', len(X_train.columns))

embeded_lgb_support = embeded_lgb_selector.get_support()

# print(embeded_lgb_support)

embeded_lgb_feature = X_train.loc[:,embeded_lgb_support].columns.tolist()

print('end cols len after feature selection: ', len(embeded_lgb_feature))

print(str(len(embeded_lgb_feature)), 'selected features')
train, test = reload()

print(f'len train: {len(train)}, len_test: {len(test)}')

train, test = reduce_mem_usage(train), reduce_mem_usage(test)
train = reduce_mem_usage(clip(train))

train = add_df_features(train.drop(columns='matchType'))

test = reduce_mem_usage(clip(test))

test = add_df_features(test.drop(columns='matchType'))

print(f'len train: {len(train)}, len_test: {len(test)}')

gc.collect()
# Select only most important features (115 in this case)

train = train[embeded_lgb_feature]

test = test[embeded_lgb_feature]

gc.collect()
# Fit LightGbm predictor

import lightgbm as lgb

params={'learning_rate': 0.1,

        'objective':'mae',

        'metric':'mae',

        'num_leaves': 31,

        'verbose': 1,

        'random_state':42,

        'bagging_fraction': 0.7,

        'feature_fraction': 0.7

       }



reg = lgb.LGBMRegressor(**params, n_estimators=200)

X_train, y_train = get_X_Y_fromdf(train)

reg.fit(X_train, y_train)

X_test, _ = get_X_Y_fromdf(test)

pred = reg.predict(X_test, num_iteration=reg.best_iteration_)
# print(len(df_sub))

print(len(test))

print(len(X_test))

df_sub = pd.read_csv(filesdir + "sample_submission_V2.csv")

df_test = pd.read_csv(filesdir + "test_V2.csv")

df_sub['winPlacePerc'] = pred

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