import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import gc
import os

sns.set_style('darkgrid')
sns.set_palette('bone')

pd.options.display.float_format = '{:.5g}'.format

print(os.listdir("../input"))
def toTapleList(list1,list2):
    return list(itertools.product(list1,list2))
# Thanks and credited to https://www.kaggle.com/gemartin
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
%%time
train = pd.read_csv('../input/train_V2.csv')
train = reduce_mem_usage(train)
test = pd.read_csv('../input/test_V2.csv')
test = reduce_mem_usage(test)
print(train.shape, test.shape)
train.info()
null_cnt = train.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])
# dropna
train.dropna(inplace=True)
train.describe(include=np.number).drop('count').T
#plt.subplots(figsize=(16, 9))
#sns.heatmap(train.corr(), annot=True, linewidths=.6, fmt='.2f', vmax=1, vmin=-1, center=0, cmap='Blues')
for c in ['Id','groupId','matchId']:
    print(f'unique [{c}] count:', train[c].nunique())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))

train['matchType'].value_counts().plot.bar(ax=ax[0])

'''
solo  <-- solo,solo-fpp,normal-solo,normal-solo-fpp
duo   <-- duo,duo-fpp,normal-duo,normal-duo-fpp,crashfpp,crashtpp
squad <-- squad,squad-fpp,normal-squad,normal-squad-fpp,flarefpp,flaretpp
'''
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
train['matchType'].apply(mapper).value_counts().plot.bar(ax=ax[1])
for q in ['numGroups == maxPlace','numGroups != maxPlace']:
    print(q, ':', len(train.query(q)))
# describe
cols = ['numGroups','maxPlace']
desc1 = train.groupby('matchType')[cols].describe()[toTapleList(cols,['min','mean','max'])]
# groups in match
group = train.groupby(['matchType','matchId','groupId']).count().groupby(['matchType','matchId']).size().to_frame('groups in match')
desc2 = group.groupby('matchType').describe()[toTapleList(['groups in match'],['min','mean','max'])]

pd.concat([desc1, desc2], axis=1)
# players in match, group
# 100 players join the same server,
# so in the case of duos the max teams are 50 and in the case of squads the max teams are 25.
match = train.groupby(['matchType','matchId']).size().to_frame('players in match')
group = train.groupby(['matchType','matchId','groupId']).size().to_frame('players in group')
pd.concat([match.groupby('matchType').describe()[toTapleList(['players in match'],['min','mean','max'])], 
           group.groupby('matchType').describe()[toTapleList(['players in group'],['min','mean','max'])]], axis=1)
print(group['players in group'].nlargest(5))
del match,group
# Too many group players
# ex) matchId=='41a634f62f86b7', groupId=='128b07271aa012'
subset = train[train['matchId']=='41a634f62f86b7']
sub_grp = subset[subset['groupId']=='128b07271aa012']

print('matchId==\'41a634f62f86b7\' & groupId==\'128b07271aa012\'')
print('-'*50)
print('players:',len(subset))
print('groups:',subset['groupId'].nunique())
print('numGroups:',subset['numGroups'].unique())
print('maxPlace:',subset['maxPlace'].unique())
print('-'*50)
print('max-group players:',len(sub_grp))
print('max-group winPlacePerc:',sub_grp['winPlacePerc'].unique())
print('-'*50)
print('winPlacePerc:',subset['winPlacePerc'].sort_values().unique())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
# there are two types of maps?
train['matchDuration'].hist(bins=50, ax=ax[0])
train.query('matchDuration >= 1400 & matchDuration <= 1800')['matchDuration'].hist(bins=50, ax=ax[1])
train[train['matchDuration'] == train['matchDuration'].min()].head()
train[train['matchDuration'] == train['matchDuration'].max()].head()
# same match is same duration
train.groupby('matchId').agg({'matchDuration':'var'}).fillna(0).query('matchDuration != 0')
fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

cols = ['boosts','heals','revives']
for col, ax in zip(cols, ax.ravel()): 
    train.groupby(col).mean()['winPlacePerc'].plot.line(ax=ax)
# solo players are not revives
'solo' in train.query('revives > 0')['matchType'].unique()
train.groupby(['matchType'])['killPlace'].describe()[['min','mean','max']]
''' important 
killPlace is a sorted ranking of kills and winPlacePerc in each match.
'''
sub = train[train['matchId'] == train['matchId'].min()].sort_values(['winPlacePerc','killPlace'])
cols = ['groupId','kills','winPlacePerc','killPlace']
sub[cols]
col = 'killPlace'
sub = train[['winPlacePerc',col]].copy()
sub[col] = pd.cut(sub[col], [10*x for x in range(0,11)], right=False)
sub.groupby(col).mean()['winPlacePerc'].plot.bar()
fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

cols = ['kills','killStreaks','DBNOs']
for col, ax in zip(cols, ax.ravel()):
    sub = train[['winPlacePerc',col]].copy()
    sub[col] = pd.cut(sub[col], 6)
    #sns.barplot(x=col, y='winPlacePerc', data=sub.groupby(col, as_index=False).mean(), ax=ax)
    sub.groupby(col).mean()['winPlacePerc'].plot.bar(ax=ax)
fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

cols = ['headshotKills','roadKills','teamKills']
for col, ax in zip(cols, ax.ravel()): 
    sub = train[['winPlacePerc',col]].copy()
    sub.loc[sub[col] >= 5, col] = '5+'  
    sub[col] = sub[col].astype(str)
    sub.groupby(col).mean()['winPlacePerc'].plot.bar(ax=ax)
#train[train['kills'] != 0].plot.scatter(x='longestKill', y='winPlacePerc')
col = 'longestKill'
sub = train[['winPlacePerc',col]].copy()
sub[col] = pd.cut(sub[col], 6)
sub.groupby(col).mean()['winPlacePerc'].plot.bar()
fig, ax = plt.subplots(1, 3, figsize=(16, 4))

cols = ['walkDistance', 'rideDistance', 'swimDistance']
for col, ax in zip(cols, ax.ravel()):
    sub = train[['winPlacePerc',col]].copy()
    sub[col] = pd.cut(sub[col], 6)
    sub.groupby(col).mean()['winPlacePerc'].plot.bar(ax=ax)
# cheater?
sq = ''
querys = ['walkDistance == 0 & rideDistance == 0 & swimDistance == 0',' & kills > 0',' & headshotKills > 0',' & heals > 0']
for q in querys:
    sq += q
    sub = train.query(sq)
    print(sq, '\n count:', len(sub), ' winPlacePerc:', sub['winPlacePerc'].mean())
del sub
sub = train[['walkDistance','rideDistance','swimDistance','winPlacePerc']].copy()
walk = train['walkDistance']
sub['walkDistanceBin'] = pd.cut(walk, [0, 0.001, walk.quantile(.25), walk.quantile(.5), walk.quantile(.75), 99999])
sub['rideDistanceBin'] = (train['rideDistance'] > 0).astype(int)
sub['swimDistanceBin'] = (train['swimDistance'] > 0).astype(int)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
sub.groupby('walkDistanceBin').mean()['winPlacePerc'].plot.bar(ax=ax[0])
sub.groupby('rideDistanceBin').mean()['winPlacePerc'].plot.bar(ax=ax[1])
sub.groupby('swimDistanceBin').mean()['winPlacePerc'].plot.bar(ax=ax[2])
del sub, walk
fig, ax = plt.subplots(1, 3, figsize=(16, 3), sharey=True)

cols = ['killPoints','rankPoints','winPoints']
for col, ax in zip(cols, ax.ravel()): 
    train.plot.scatter(x=col, y='winPlacePerc', ax=ax)
# rankPoint: being deprecated
# killPoints,winPoints: If there is a value other than -1 in rankPoints, then any 0 should be treated as a “None”.
cols = ['killPoints','rankPoints','winPoints']
querys = ['rankPoints <= 0 & killPoints != 0', 'rankPoints <= 0 & winPoints != 0',
          'killPoints == 0 & rankPoints != 0', 'killPoints == 0 & winPoints != 0',
          'winPoints == 0 & rankPoints != 0', 'winPoints == 0 & killPoints != 0']
for q in querys:
    print('count (%s):' % q, len(train[cols].query(q)))
#sns.distplot(train['killPoints'], kde=False)
train['winPlacePerc'].describe()
# confirm unique winPlace in group
#nuniquePlace = train.groupby(['matchId','groupId'])['winPlacePerc'].nunique()
#print('not unique winPlace in group:', len(nuniquePlace[nuniquePlace > 1]))
#del nuniquePlace
print('match count:', train['matchId'].nunique())

# not contains 1st place
maxPlacePerc = train.groupby('matchId')['winPlacePerc'].max()
print('match [not contains 1st place]:', len(maxPlacePerc[maxPlacePerc != 1]))
del maxPlacePerc

# edge case
edge = train[(train['maxPlace'] > 1) & (train['numGroups'] == 1)]
print('match [maxPlace>1 & numGroups==1]:', len(edge.groupby('matchId')))
print(' - unique winPlacePerc:', edge['winPlacePerc'].unique())
pd.concat([train[train['winPlacePerc'] == 1].head(5),
           train[train['winPlacePerc'] == 0].head(5)],
          keys=['winPlacePerc_1', 'winPlacePerc_0'])
all_data = train.append(test, sort=False).reset_index(drop=True)
del train, test
gc.collect()
match = all_data.groupby('matchId')
all_data['killPlacePerc'] = match['kills'].rank(pct=True).values
distance = (all_data['rideDistance'] + all_data['walkDistance'] + all_data['swimDistance'])

all_data['zombi'] = ((distance == 0) & (all_data['kills'] == 0)
                     & (all_data['weaponsAcquired'] == 0) 
                     & (all_data['matchType'].str.contains('solo'))).astype(int)
all_data['cheater'] = ((all_data['kills'] / distance >= 1)
                       | (all_data['kills'] > 30) | (all_data['roadKills'] > 10)).astype(int)
del distance
pd.concat([all_data['zombi'].value_counts(), all_data['cheater'].value_counts()], axis=1).T
#all_data.loc[all_data['rankPoints'] <= 0, 'rankPoints'] = all_data.loc[all_data['rankPoints'] <= 0, 'killPoints']
#all_data.loc[all_data['killPoints'] == 0, 'killPoints'] = all_data.loc[all_data['killPoints'] == 0, 'rankPoints']
#all_data.loc[all_data['winPoints'] == 0, 'winPoints'] = all_data.loc[all_data['winPoints'] == 0, 'rankPoints']
all_data.drop(['rankPoints','killPoints','winPoints'], axis=1, inplace=True)
def fillInf(df, val):
    numcols = df.select_dtypes(include='number').columns
    cols = numcols[numcols != 'winPlacePerc']
    df[df == np.Inf] = np.NaN
    df[df == np.NINF] = np.NaN
    for c in cols: df[c].fillna(val, inplace=True)
all_data['_totalDistance'] = all_data['rideDistance'] + all_data["walkDistance"] + all_data["swimDistance"]
all_data["_specialKills"] = all_data["headshotKills"] + all_data["roadKills"]
all_data['_healthItems'] = all_data['heals'] + all_data['boosts']

all_data['_headshotKillRate'] = all_data['headshotKills'] / all_data['kills']
all_data['_killsOverWalkDistance'] = all_data['kills'] / all_data['walkDistance']
all_data['_killsOverDistance'] = all_data['kills'] / all_data['_totalDistance']
all_data['_killPlaceOverMaxPlace'] = all_data['killPlace'] / all_data['maxPlace']
#all_data['_distanceOverWeapons'] = all_data['_totalDistance'] / all_data['weaponsAcquired']
#all_data['_distanceOverHeals'] = all_data['_totalDistance'] / all_data['_healthItems']

# suicide: solo and teamKills > 0
#all_data['_suicide'] = ((all_data['member'] == 1) & (all_data['teamKills'] > 0)).astype(int)

fillInf(all_data, 0)
agg_col = list(all_data.columns)
exclude_agg_col = ['Id','matchId','groupId','matchType','matchDuration','maxPlace','numGroups','winPlacePerc']
for c in exclude_agg_col:
    agg_col.remove(c)
    
print(agg_col)
def grouping(df):
    group = df.groupby(['matchId','groupId','matchType'])
    # group count, mean, max, min
    groupCount = group.size().to_frame('member')
    groupMean = group.mean()
    groupMax = group[agg_col].max().rename(columns=lambda s: '_max.' + s)
    groupMin = group[agg_col].min().rename(columns=lambda s: '_min.' + s)
    return pd.concat([groupCount, groupMean, groupMax, groupMin], axis=1).reset_index()

all_data = grouping(all_data)
all_data = reduce_mem_usage(all_data)
print(all_data.shape)
cols = np.r_[agg_col,['matchId']]
match = all_data[cols].groupby('matchId')
''' match rank
'''
matchRank = match.rank(pct=True).rename(columns=lambda s: '_rank.' + s)
all_data = reduce_mem_usage(pd.concat([all_data, matchRank], axis=1))
print(all_data.shape)

del matchRank
gc.collect()
''' percentage of match summary
'''
#cols = ['member','zombi','cheater','kills','assists','walkDistance','_healthItems']
cols = agg_col
matchSum = match[cols].sum().rename(columns=lambda s: '_sum.' + s).reset_index()
all_data = pd.merge(all_data, matchSum)
print(all_data.shape)

del matchSum
gc.collect()

for c in cols:
    all_data['_percSum.' + c] = all_data[c] / all_data['_sum.' + c]
    
fillInf(all_data, 0)
''' percentage of match mean
'''
#cols = ['damageDealt','kills','assists','walkDistance','_healthItems','_killPlaceOverMaxPlace']
cols = agg_col
matchMean = match[cols].mean().rename(columns=lambda s: '_matchMean.' + s).reset_index()
all_data = pd.merge(all_data, reduce_mem_usage(matchMean))
print(all_data.shape)

del matchMean
gc.collect()

for c in cols:
    all_data['_percMean.' + c] = all_data[c] / all_data['_matchMean.' + c]
    all_data.drop(['_matchMean.' + c], axis=1, inplace=True)

fillInf(all_data, 0)
''' max.killPlace rank of group and min-kills
'''
killMinorRank = all_data[['matchId','_min.kills','_max.killPlace']].copy()
killMinorRank['_rank.minor.killPlace'] = killMinorRank.groupby(['matchId','_min.kills']).rank().values
all_data = pd.merge(all_data, killMinorRank)

del killMinorRank
gc.collect()
null_cnt = all_data.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])

all_data.head()
mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
#mapper = lambda x: 'solo' if ('solo' in x) else 'team'
all_data['matchTypeCat'] = all_data['matchType'].map(mapper)

print(all_data['matchTypeCat'].value_counts())
# drop constant column
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)
#all_data.drop([],axis=1,inplace=True)

cols = [col for col in all_data.columns if col not in ['Id','matchId','groupId']]
for i, t in all_data.loc[:, cols].dtypes.iteritems():
    if t == object:
        all_data[i] = pd.factorize(all_data[i])[0]
X_train = all_data[all_data['winPlacePerc'].notnull()].reset_index(drop=True)
X_test = all_data[all_data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)
del all_data
gc.collect()

Y_train = X_train.pop('winPlacePerc')
X_test_grp = X_test[['matchId','groupId']].copy()
#X_train_matchId = X_train['matchId']

# drop matchId,groupId
X_train.drop(['matchId','groupId'], axis=1, inplace=True)
X_test.drop(['matchId','groupId'], axis=1, inplace=True)

print(X_train.shape, X_test.shape)
import sys
print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],
                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import minmax_scale
import lightgbm as lgb

params={'learning_rate': 0.05,
        'objective':'mae',
        'metric':'mae',
        'num_leaves': 31,
        'verbose': 0,
        'random_state':42,
        'bagging_fraction': 0.7,
        'feature_fraction': 0.7
       }

mts = list()
fis = list()
pred = np.zeros(X_test.shape[0])
for mt in X_train['matchTypeCat'].unique():
    idx = X_train[X_train['matchTypeCat'] == mt].index
    reg = lgb.LGBMRegressor(**params, n_estimators=1000)
    reg.fit(X_train.loc[idx], Y_train.loc[idx])

    idx = X_test[X_test['matchTypeCat'] == mt].index
    pred[idx] = reg.predict(X_test.loc[idx], num_iteration=reg.best_iteration_)
    mts.append(mt)
    fis.append(reg.feature_importances_)
for mt, feature_importance in zip(mts, fis): 
    # Plot feature importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    sorted_idx = sorted_idx[len(feature_importance) - 30:]
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(12,6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_train.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance [matchTypeCat:' + str(mt) + ']')
    plt.show()
X_test_grp['winPlacePerc'] = pred

group = X_test_grp.groupby(['matchId'])
X_test_grp['winPlacePerc'] = pred
X_test_grp['_rank.winPlacePerc'] = group['winPlacePerc'].rank(method='min')
X_test = pd.concat([X_test, X_test_grp], axis=1)

sub_match = X_test_grp[['matchId','_rank.winPlacePerc']].groupby(['matchId'])
sub_group = group.count().reset_index()['matchId'].to_frame()
#sub_group['_diffRank'] = sub_match['_rank.winPlacePerc'].max().values - sub_match['_rank.winPlacePerc'].min().values

X_test = pd.merge(X_test, sub_group)
# normalize
X_test['winPlacePerc'] = X_test.groupby('matchId')['winPlacePerc'].transform(lambda x: minmax_scale(x.astype(float)))
X_test['winPlacePerc'].describe()
fullgroup = (X_test['numGroups'] == X_test['maxPlace'])

# full group (201366) --> calculate from rank
subset = X_test.loc[fullgroup]
X_test.loc[fullgroup, 'winPlacePerc'] = (subset['_rank.winPlacePerc'].values - 1) / (subset['maxPlace'].values - 1)

# not full group (684872) --> align with maxPlace
subset = X_test.loc[~fullgroup]
gap = 1.0 / (subset['maxPlace'].values - 1)
new_perc = np.around(subset['winPlacePerc'].values / gap) * gap  # half&up
X_test.loc[~fullgroup, 'winPlacePerc'] = new_perc
X_test.loc[~fullgroup, '_pred.winPlacePerc'] = np.around(X_test.loc[~fullgroup, 'winPlacePerc'].values / gap) + 1
_='''
'''
X_test.loc[~fullgroup & (X_test['matchId'] == '000b598b79aa5e'),
           ['matchId','groupId','winPlacePerc','maxPlace','numGroups','_pred.winPlacePerc','_rank.winPlacePerc']
          ].sort_values(['matchId','_pred.winPlacePerc'])
_='''
subset = X_test.loc[~fullgroup]
subset = subset.groupby(['matchId','_pred.winPlacePerc']).filter(lambda x: len(x)>1)

rank1p, rank1m = list(), list()
for n, df in subset.groupby(['matchId','_pred.winPlacePerc']):
    ranks = subset[subset['matchId'] == n[0]]['_rank.winPlacePerc'].values
    r = n[1]
    df = df.sort_values(['_rank.winPlacePerc'])
    if (r > 1) and (r < subset['maxPlace'].values[0]):
        if (r + 1) not in ranks:
            rank1p.append(df.index[len(df)-1])
            if (r - 1) not in ranks and (len(df) > 2):
                rank1m.append(df.index[0])
        elif (r - 1) not in ranks:
            rank1m.append(df.index[0])
            
X_test.loc[rank1p, '_rank.winPlacePerc'] = X_test.loc[rank1p, '_rank.winPlacePerc'] + 1
X_test.loc[rank1m, '_rank.winPlacePerc'] = X_test.loc[rank1m, '_rank.winPlacePerc'] - 1
print(len(rank1p),len(rank1m))

gap = 1.0 / (X_test.loc[rank1p, 'maxPlace'].values - 1)
X_test.loc[rank1p, 'winPlacePerc'] = (X_test.loc[rank1p, '_rank.winPlacePerc'] - 1) * gap
gap = 1.0 / (X_test.loc[rank1m, 'maxPlace'].values - 1)
X_test.loc[rank1m, 'winPlacePerc'] = (X_test.loc[rank1m, '_rank.winPlacePerc'] - 1) * gap

X_test['winPlacePerc'] = X_test['winPlacePerc'].clip(upper=1)
'''
# edge cases
X_test.loc[X_test['maxPlace'] == 0, 'winPlacePerc'] = 0
X_test.loc[X_test['maxPlace'] == 1, 'winPlacePerc'] = 1  # nothing
X_test.loc[(X_test['maxPlace'] > 1) & (X_test['numGroups'] == 1), 'winPlacePerc'] = 0
X_test['winPlacePerc'].describe()
test = pd.read_csv('../input/test_V2.csv')

submission = pd.merge(test, X_test[['matchId','groupId','winPlacePerc']])
submission = submission[['Id','winPlacePerc']]
submission.to_csv("submission.csv", index=False)
