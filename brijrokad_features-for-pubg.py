import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

import gc

import os

import sys
def toTapleList(list1,list2):

    return list(itertools.product(list1,list2))




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
train = pd.read_csv('../input/train_V2.csv')

train = reduce_mem_usage(train)

test = pd.read_csv('../input/test_V2.csv')

test = reduce_mem_usage(test)

print(train.shape, test.shape)
null_cnt = train.isnull().sum().sort_values()

print('null count:', null_cnt[null_cnt > 0])

# dropna

train.dropna(inplace=True)
train.describe(include=np.number).drop('count').T
for c in ['Id','groupId','matchId']:

    print(f'unique [{c}] count:', train[c].nunique())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))



train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[0])



'''

solo  <-- solo,solo-fpp,normal-solo,normal-solo-fpp

duo   <-- duo,duo-fpp,normal-duo,normal-duo-fpp,crashfpp,crashtpp

squad <-- squad,squad-fpp,normal-squad,normal-squad-fpp,flarefpp,flaretpp

'''

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

train['matchType'] = train['matchType'].apply(mapper)

train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[1])
all_data = train.append(test, sort=False).reset_index(drop=True)

del train, test

gc.collect()
match = all_data.groupby('matchId')

all_data['killsPerc'] = match['kills'].rank(pct=True).values

all_data['killPlacePerc'] = match['killPlace'].rank(pct=True).values

all_data['walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

#all_data['damageDealtPerc'] = match['damageDealt'].rank(pct=True).values

all_data['walkPerc_killsPerc'] = all_data['walkDistancePerc'] / all_data['killsPerc']
all_data['_totalDistance'] = all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance']
def fillInf(df, val):

    numcols = df.select_dtypes(include='number').columns

    cols = numcols[numcols != 'winPlacePerc']

    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN

    for c in cols: df[c].fillna(val, inplace=True)
all_data['_healthItems'] = all_data['heals'] + all_data['boosts']

all_data['_headshotKillRate'] = all_data['headshotKills'] / all_data['kills']

all_data['_killPlaceOverMaxPlace'] = all_data['killPlace'] / all_data['maxPlace']

all_data['_killsOverWalkDistance'] = all_data['kills'] / all_data['walkDistance']



fillInf(all_data, 0)
match = all_data.groupby(['matchId'])

group = all_data.groupby(['matchId','groupId','matchType'])



# target feature (max, min)

agg_col = list(all_data.columns)

exclude_agg_col = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']

for c in exclude_agg_col:

    agg_col.remove(c)

print(agg_col)



# target feature (sum)

sum_col = ['kills','killPlace','damageDealt','walkDistance','_healthItems']
match_data = pd.concat([

    match.size().to_frame('m.players'), 

    match[sum_col].sum().rename(columns=lambda s: 'm.sum.' + s), 

    match[sum_col].max().rename(columns=lambda s: 'm.max.' + s),

    match[sum_col].mean().rename(columns=lambda s: 'm.mean.' + s)

    ], axis=1).reset_index()

match_data = pd.merge(match_data, 

    group[sum_col].sum().rename(columns=lambda s: 'sum.' + s).reset_index())

match_data = reduce_mem_usage(match_data)



print(match_data.shape)
minKills = all_data.sort_values(['matchId','groupId','kills','killPlace']).groupby(

    ['matchId','groupId','kills']).first().reset_index().copy()

for n in np.arange(4):

    c = 'kills_' + str(n) + '_Place'

    nKills = (minKills['kills'] == n)

    minKills.loc[nKills, c] = minKills[nKills].groupby(['matchId'])['killPlace'].rank().values

    match_data = pd.merge(match_data, minKills[nKills][['matchId','groupId',c]], how='left')

    #match_data[c].fillna(0, inplace=True)

match_data = reduce_mem_usage(match_data)

del minKills, nKills



print(match_data.shape)
all_data = pd.concat([

    group.size().to_frame('players'),

    group.mean(),

    group[agg_col].max().rename(columns=lambda s: 'max.' + s),

    group[agg_col].min().rename(columns=lambda s: 'min.' + s),

    ], axis=1).reset_index()

all_data = reduce_mem_usage(all_data)



print(all_data.shape)
numcols = all_data.select_dtypes(include='number').columns.values

numcols = numcols[numcols != 'winPlacePerc']
all_data = pd.merge(all_data, match_data)

del match_data

gc.collect()



all_data['enemy.players'] = all_data['m.players'] - all_data['players']

for c in sum_col:

    #all_data['enemy.' + c] = (all_data['m.sum.' + c] - all_data['sum.' + c]) / all_data['enemy.players']

    #all_data['p.sum_msum.' + c] = all_data['sum.' + c] / all_data['m.sum.' + c]

    #all_data['p.max_mmean.' + c] = all_data['max.' + c] / all_data['m.mean.' + c]

    all_data['p.max_msum.' + c] = all_data['max.' + c] / all_data['m.sum.' + c]

    all_data['p.max_mmax.' + c] = all_data['max.' + c] / all_data['m.max.' + c]

    all_data.drop(['m.sum.' + c, 'm.max.' + c], axis=1, inplace=True)

    

fillInf(all_data, 0)

print(all_data.shape)
match = all_data.groupby('matchId')

matchRank = match[numcols].rank(pct=True).rename(columns=lambda s: 'rank.' + s)

all_data = reduce_mem_usage(pd.concat([all_data, matchRank], axis=1))

rank_col = matchRank.columns

del matchRank

gc.collect()



# instead of rank(pct=True, method='dense')

match = all_data.groupby('matchId')

matchRank = match[rank_col].max().rename(columns=lambda s: 'max.' + s).reset_index()

all_data = pd.merge(all_data, matchRank)

for c in numcols:

    all_data['rank.' + c] = all_data['rank.' + c] / all_data['max.rank.' + c]

    all_data.drop(['max.rank.' + c], axis=1, inplace=True)

del matchRank

gc.collect()



print(all_data.shape)
killMinorRank = all_data[['matchId','min.kills','max.killPlace']].copy()

group = killMinorRank.groupby(['matchId','min.kills'])

killMinorRank['rank.minor.maxKillPlace'] = group.rank(pct=True).values

all_data = pd.merge(all_data, killMinorRank)



killMinorRank = all_data[['matchId','max.kills','min.killPlace']].copy()

group = killMinorRank.groupby(['matchId','max.kills'])

killMinorRank['rank.minor.minKillPlace'] = group.rank(pct=True).values

all_data = pd.merge(all_data, killMinorRank)



del killMinorRank

gc.collect()
# drop constant column

constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]

print('drop columns:', constant_column)

all_data.drop(constant_column, axis=1, inplace=True)
all_data['matchType'] = all_data['matchType'].apply(mapper)



all_data = pd.concat([all_data, pd.get_dummies(all_data['matchType'])], axis=1)

all_data.drop(['matchType'], axis=1, inplace=True)



all_data['matchId'] = all_data['matchId'].apply(lambda x: int(x,16))

all_data['groupId'] = all_data['groupId'].apply(lambda x: int(x,16))
null_cnt = all_data.isnull().sum().sort_values()

print(null_cnt[null_cnt > 0])
#all_data.drop([],axis=1,inplace=True)



cols = [col for col in all_data.columns if col not in ['Id','matchId','groupId']]

for i, t in all_data.loc[:, cols].dtypes.iteritems():

    if t == object:

        all_data[i] = pd.factorize(all_data[i])[0]



all_data = reduce_mem_usage(all_data)

all_data.head()
dfTrain = pd.DataFrame(all_data)



dfTrain.to_csv('train_V4.csv', index = False)