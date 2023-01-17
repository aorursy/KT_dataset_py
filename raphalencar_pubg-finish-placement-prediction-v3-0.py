import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

import tensorflow as tf

import itertools

import sys

import gc

from IPython import get_ipython
# setup

%matplotlib inline

pd.options.display.max_columns = 50



sns.set_style('darkgrid')

sns.set_palette('bone')



warnings.filterwarnings('ignore')
def toTupleList(list1, list2):

    return list(itertools.product(list1, list2))
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
def fillInf(df, val):

    numcols = df.select_dtypes(include='number').columns

    cols = numcols[numcols != 'winPlacePerc']

    df[df == np.Inf] = np.NaN

    df[df == np.NINF] = np.NaN

    for c in cols: df[c].fillna(val, inplace=True)
%%time 

train = pd.read_csv('../input/train_V2.csv')

train = reduce_mem_usage(train)



test = pd.read_csv('../input/test_V2.csv')

test = reduce_mem_usage(test)



print(train.shape, test.shape)
train.info()
nullValues = train[train.isnull().any(axis=1)]

nullValues
print('null values:', len(nullValues))



# let's dropna

train.dropna(inplace=True)
train.shape
train.describe().drop('count').T
train['matchType'].value_counts()
numCols = train.select_dtypes(include=np.number).columns



fig, ax = plt.subplots(5, 5, figsize=(15, 15))

for c, ax in zip(list(numCols), ax.ravel()):

    ax.set_xlabel(c)

    train[c].hist(bins=10, ax=ax)

    

plt.tight_layout()
corr_matrix = train.corr() # creating correlation matrix

corr_matrix['winPlacePerc'].sort_values(ascending=False)
sns.jointplot(x='winPlacePerc', y='walkDistance', data=train, color='red', size=5)

plt.show()
sns.jointplot(x='winPlacePerc', y='boosts', data=train, color='blue', size=5)

plt.show()
sns.jointplot(x='winPlacePerc', y='weaponsAcquired', data=train, color='lime', size=5)

plt.show()
sns.jointplot(x='winPlacePerc', y='killPlace', data=train, color='yellow', size=5)

plt.show()
fig, ax = plt.subplots(figsize=(20,20))

sns.heatmap(data=corr_matrix, annot=True, fmt='.2f', vmin=0, vmax=1, ax=ax)

plt.show()
sns.set()

rows=1000000

cols = ['winPlacePerc', 'walkDistance', 'boosts', 'weaponsAcquired', 'damageDealt', 'killPlace']

sns.pairplot(train[cols][:rows], size=2.5)

plt.show()
data = train.copy()

data = data[data['heals'] < data['heals'].quantile(0.99)]

data = data[data['boosts'] < data['boosts'].quantile(0.99)]



fig, ax = plt.subplots(figsize=(20,10))

sns.pointplot(x='boosts', y='winPlacePerc', data=data, color='red', alpha=0.4)

sns.pointplot(x='heals', y='winPlacePerc', data=data, color='blue', alpha=0.4)

plt.text(10, 0.7, 'Boosts', color='red', fontsize=14)

plt.text(10, 0.5, 'Heals', color='blue', fontsize=14)

plt.xlabel('Heal/Boost amount', fontsize=15, color='blue')

plt.ylabel('winPlacePerc', fontsize=15, color='blue')

plt.title('Boosts vs Heals', fontsize=20, color='blue')

plt.grid()

plt.show()
for c in ['Id', 'groupId', 'matchId']:

    print(f'{c} different values:', train[c].nunique())
mapper = lambda x: 'solo' if('solo' in x) else 'duo' if('duo' in x) or ('crash' in x) else 'squad'



train['matchType'] = train['matchType'].apply(mapper)

train.groupby('matchId')['matchType'].first().value_counts().plot.bar()

plt.show()
for q in ['maxPlace == numGroups', 'maxPlace != numGroups']:

    print(q, ':', len(train.query(q)))
'''matchId: aeb375fc57110c

matchType: squad

numGroups: 25

maxPlace: 26'''

# df = train.query('matchId == "aeb375fc57110c"')

# df.groupby(['matchType', 'groupId']).size()
cols = ['maxPlace', 'numGroups']

desc = train.groupby('matchType')[cols].describe()[toTupleList(cols, ['min', 'mean', 'max'])]



# groups in match

group = train.groupby(['matchType', 'matchId', 'groupId']).count().groupby(['matchType', 'matchId']).size().to_frame('groups in match')

descGroup = group.groupby('matchType').describe()[toTupleList(['groups in match'], ['min', 'mean', 'max'])]



pd.concat([desc, descGroup], axis=1)
match = train.groupby(['matchId', 'matchType']).size().to_frame('players in match')

group = train.groupby(['matchId', 'matchType', 'groupId']).size().to_frame('players in group')



descMatch = match.groupby('matchType').describe()[toTupleList(['players in match'], ['min', 'mean', 'max'])]

descGroup = group.groupby('matchType').describe()[toTupleList(['players in group'], ['min', 'mean', 'max'])]



pd.concat([descMatch, descGroup], axis=1)
group = train.groupby(['matchId','matchType', 'groupId'])['Id'].count().to_frame('players').reset_index()

group.loc[group['players'] > 4, 'players'] = '5+'

group['players'] = group['players'].astype(str)



fig, ax = plt.subplots(1, 3, figsize=(16, 4))

for mt, ax in zip(['solo', 'duo', 'squad'], ax.ravel()):

    ax.set_xlabel(mt)

    group[group['matchType'] == mt]['players'].value_counts().sort_index().plot.bar(ax=ax)
solo = train['matchType'].str.contains('solo')



soloSummary = train.loc[solo].groupby('matchId')['kills'].sum().describe()

teamSummary = train.loc[~solo].groupby('matchId')['kills'].sum().describe()



pd.concat([soloSummary, teamSummary], keys=['solo', 'team'], axis=1).T
soloSummary = train.loc[solo].groupby('matchId')['assists'].sum().describe()

teamSummary = train.loc[~solo].groupby('matchId')['assists'].sum().describe()



pd.concat([soloSummary, teamSummary], keys=['solo', 'team'], axis=1).T
sign = lambda x: 'p<=0' if x <= 0 else 'p>0'

rankWin = pd.crosstab(train['rankPoints'].apply(sign), train['winPoints'].apply(sign), margins=False)

rankKill = pd.crosstab(train['rankPoints'].apply(sign), train['killPoints'].apply(sign), margins=False)



pd.concat([rankWin, rankKill], keys=['winPoints, killPoints'], axis=1)
winPlacePerc_1 = train[train['winPlacePerc'] == 1].head()

winPlacePerc_0 = train[train['winPlacePerc'] == 0].head()



pd.concat([winPlacePerc_1, winPlacePerc_0], keys=['winPlacePerc_1', 'winPlacePerc_0'])
train[['winPlacePerc']].describe().drop('count').T
cols = ['kills','teamKills','DBNOs','revives','assists','boosts','heals','damageDealt',

    'walkDistance','rideDistance','swimDistance','weaponsAcquired']



aggs = ['count', 'min', 'mean', 'max']



# solo match

solo = train['matchType'].str.contains('solo')

soloMatch = train.loc[solo].groupby('matchId')[cols].sum()



# team match

teamMatch = train.loc[~solo].groupby('matchId')[cols].sum()



pd.concat([soloMatch.describe().T[aggs], teamMatch.describe().T[aggs]], keys=['solo', 'team'], axis=1)
# data['numPlayers'] = data.groupby('matchId')['matchId'].transform('count')

# data['numPlayers'].head()
# plt.figure(figsize=(15,10))

# sns.countplot(data.query('numPlayers >= 50')['numPlayers'])

# plt.show()
# train_data['killsNorm'] = train_data['kills'] * ((100 - train_data['numPlayers']) / 100 + 1)

# train_data['damageDealt'] = train_data['damageDealt'] * ((100 - train_data['numPlayers']) / 100 + 1)

# train_data['maxPlace'] = train_data['maxPlace'] * ((100 - train_data['numPlayers']) / 100 + 1)

# train_data['matchDuration'] = train_data['matchDuration'] * ((100 - train_data['numPlayers']) / 100 + 1)
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']

train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

train.query('killsWithoutMoving == True').shape
train.query('killsWithoutMoving == True').head()
train.drop(train.query('killsWithoutMoving == True').index, inplace=True)

train.shape
train.drop('killsWithoutMoving', axis=1, inplace=True)
sns.boxplot(y='roadKills', data=train, color='blue')

plt.show()
plt.figure(figsize=(12,4))

sns.distplot(train['roadKills'], bins=10)

plt.show()
train['roadKills'].mean()
train.query('roadKills > 10').shape
train.query('roadKills > 10').head()
train.query('Id == "c3e444f7d1289f"')
train.drop(train.query('Id == "c3e444f7d1289f"').index, inplace=True)
train.shape
sns.boxplot(y='weaponsAcquired', data=train, color='blue')

plt.show()
plt.figure(figsize=(12,4))

sns.distplot(train['weaponsAcquired'], bins=100)

plt.show()
train['weaponsAcquired'].mean()
train.query('weaponsAcquired > 50').shape
query = '(weaponsAcquired > 50) & (totalDistance < 500)'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
query = '(weaponsAcquired > 1) & (totalDistance == 0)'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
query = '(weaponsAcquired > 100)'

train.query(query).head()
train.drop(train.query('Id == "da68d2812229a8"').index, inplace=True)
sns.boxplot(y='walkDistance', data=train, color='blue')

plt.show()
plt.figure(figsize=(12,4))

sns.distplot(train['walkDistance'], bins=10)

plt.show()
train['walkDistance'].mean()
query = 'walkDistance > 15000'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
train.shape
sns.boxplot(y='rideDistance', data=train, color='blue')

plt.show()
plt.figure(figsize=(12,4))

sns.distplot(train['rideDistance'], bins=10)

plt.show()
train['rideDistance'].mean()
query = '(rideDistance > 30000)'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
query = '(rideDistance == totalDistance) & (weaponsAcquired >= 1)'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
sns.boxplot(y='swimDistance', data=train, color='blue')

plt.show()
plt.figure(figsize=(12,4))

sns.distplot(train['swimDistance'], bins=10)

plt.show()
train['swimDistance'].mean()
query = 'swimDistance > 3000'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
train.shape
query = '(swimDistance == totalDistance) & (weaponsAcquired >= 1)'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
train.shape
sns.boxplot(y='heals', data=train, color='yellow')

plt.show()
sns.distplot(train['heals'], bins=10)

plt.show()
train['heals'].mean()
query = 'heals > 50'

train.query(query).shape
train.query(query).head()
train.drop(train.query(query).index, inplace=True)
train = train.drop('totalDistance', axis=1)
train.columns
data = train.append(test, sort=False).reset_index(drop=True)

del train, test

gc.collect()
match = data.groupby('matchId')

data['_killsPerc'] = match['kills'].rank(pct=True).values

data['_damageDealtPerc'] = match['damageDealt'].rank(pct=True).values

data['_walkDistancePerc'] = match['walkDistance'].rank(pct=True).values

data['_walkPerc_killsPerc'] = data['_walkDistancePerc'] / data['_killsPerc']



data['_totalDistance'] = data['walkDistance'] + data['rideDistance'] + data['swimDistance']

data['_items'] = data['heals'] + data['boosts']

data['_damageDealtAndWalkDistance'] = data['damageDealt'] + data['walkDistance']

data['_headshotRate'] = data['headshotKills'] / data['kills']

data['_killPlace_maxPlace'] = data['killPlace'] / data['maxPlace']



fillInf(data, 0)
nullCnt = data.isnull().sum()

nullCnt
corr = data.corr()

corr['winPlacePerc'].sort_values(ascending=False)

del corr 

gc.collect()



depCols = ['rideDistance', 'swimDistance', 'heals', 'boosts', 'headshotKills', 'roadKills', 'vehicleDestroys', 'killStreaks', 'DBNOs', 'killPoints', 'rankPoints', 'winPoints', 'matchDuration']

data = data.drop(depCols, axis=1)
'''Grouping by match and groups'''

match = data.groupby('matchId')

group = data.groupby(['matchId', 'groupId', 'matchType'])



agg = list(data.columns)

cols_to_exclude = ['Id','matchId','groupId','matchType','maxPlace','numGroups','winPlacePerc']



for c in cols_to_exclude:

    agg.remove(c)

    

sum_col = ['kills','killPlace','damageDealt','walkDistance','_items']



matchData = pd.concat([

    match.size().to_frame('m.players'),

    match[sum_col].sum().rename(columns=lambda s: 'm.sum.' + s),

    match[sum_col].max().rename(columns=lambda s: 'm.max.' + s),

    match[sum_col].mean().rename(columns=lambda s: 'm.mean.' + s)

], axis=1).reset_index()



matchData = pd.merge(matchData, group[sum_col].sum().rename(columns=lambda s: 'sum.' + s).reset_index())

matchData = reduce_mem_usage(matchData)
minKills = data.sort_values(['matchId', 'groupId', 'kills', 'killPlace']).groupby(['matchId', 'groupId', 'kills']).first().reset_index().copy()



for n in np.arange(4):

    c = 'kills_' + str(n) + '_Place'

    nKills = (minKills['kills'] == n)

    minKills.loc[nKills, c] = minKills[nKills].groupby(['matchId'])['killPlace'].rank().values

    matchData = pd.merge(matchData, minKills[nKills][['matchId','groupId',c]], how='left')



matchData = reduce_mem_usage(matchData)

del minKills, nKills
matchData.head()
'''groups'''

data = pd.concat([

    group.size().to_frame('players'),

    group.mean(),

    group[agg].max().rename(columns=lambda s: 'max.' + s),

    group[agg].min().rename(columns=lambda s: 'min.' + s),

], axis=1).reset_index()



data = reduce_mem_usage(data)
numCols = data.select_dtypes(include='number').columns.values

numCols = numCols[numCols != 'winPlacePerc']
'''match summary'''

data = pd.merge(data, matchData)

del matchData

gc.collect()
sum_col
data['enemy.players'] = data['m.players'] - data['players']

 

for c in sum_col:

    data['p.max_msum.' + c] = data['max.' + c] / data['m.sum.' + c]

    data['p.max_mmax.' + c] = data['max.' + c] / data['m.max.' + c]

    data.drop(['m.sum.' + c, 'm.max.' + c], axis=1, inplace=True)

    

fillInf(data, 0)
data.shape
'''match rank'''

match = data.groupby('matchId')

matchRank = match[numCols].rank(pct=True).rename(columns=lambda s: 'rank.' + s)

data = reduce_mem_usage(pd.concat([data, matchRank], axis=1))

rank_col = matchRank.columns

del matchRank

gc.collect()
match = data.groupby('matchId')

matchRank = match[rank_col].max().rename(columns=lambda s: 'max.' + s).reset_index()

data = pd.merge(data, matchRank)

for c in numCols:

    data['rank.' + c] = data['rank.' + c] / data['max.rank.' + c]

    data.drop(['max.rank.' + c], axis=1, inplace=True)



del matchRank

gc.collect()
'''drop constant columns'''

constant_columns = [col for col in data.columns if data[col].nunique() == 1]

print(constant_columns)

data.drop(constant_columns, axis=1, inplace=True)
data['matchType'] = data['matchType'].apply(mapper)



data = pd.concat([data, pd.get_dummies(data['matchType'])], axis=1)

data.drop(['matchType'], axis=1, inplace=True)



data['matchId'] = data['matchId'].apply(lambda x: int(x, 16))

data['groupId'] = data['groupId'].apply(lambda x: int(x, 16))
nullCnt = data.isnull().sum().sort_values()

print(nullCnt[nullCnt > 0])
cols = [col for col in data.columns if col not in ['Id', 'matchId', 'groupId']]

for i, t in data.loc[:, cols].dtypes.iteritems():

    if t == object:

        data[i] = pd.factorize(data[i])[0]



data = reduce_mem_usage(data)

data.head()
# mType = ['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp']

# idx = train_data[~train_data['matchType'].isin(mType)].index

# train_data.drop(idx, inplace=True)

# train_data['matchType'].value_counts()
# train_data.shape
# match_type_count = train_data['matchType'].value_counts()

# sns.barplot(match_type_count.index, match_type_count.values, alpha=0.9)

# plt.show()
# train_data = pd.get_dummies(train_data, columns=['matchType'])

# train_data.head()
X_train = data[data['winPlacePerc'].notnull()].reset_index(drop=True)

X_test = data[data['winPlacePerc'].isnull()].drop(['winPlacePerc'], axis=1).reset_index(drop=True)

del data

gc.collect()
Y_train = X_train.pop('winPlacePerc')

X_test_grp = X_test[['matchId', 'groupId']].copy()

train_matchId = X_train['matchId']



X_train.drop(['matchId', 'groupId'], axis=1, inplace=True)

X_test.drop(['matchId', 'groupId'], axis=1, inplace=True)



print(X_train.shape, X_test.shape)
print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
from sklearn.model_selection import GroupKFold

from sklearn.preprocessing import minmax_scale

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



reg = lgb.LGBMRegressor(**params, n_estimators=10000)

reg.fit(X_train, Y_train)

pred = reg.predict(X_test, num_iteration=reg.best_iteration_)
# Plot feature importance

feature_importance = reg.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[len(feature_importance) - 30:]

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(12,8))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
X_train.columns[np.argsort(-feature_importance)].values
X_test_grp['_nofit.winPlacePerc'] = pred



group = X_test_grp.groupby(['matchId'])

X_test_grp['winPlacePerc'] = pred

X_test_grp['_rank.winPlacePerc'] = group['winPlacePerc'].rank(method='min')

X_test = pd.concat([X_test, X_test_grp], axis=1)
fullgroup = (X_test['numGroups'] == X_test['maxPlace'])



# full group (201366) --> calculate from rank

subset = X_test.loc[fullgroup]

X_test.loc[fullgroup, 'winPlacePerc'] = (subset['_rank.winPlacePerc'].values - 1) / (subset['maxPlace'].values - 1)



# not full group (684872) --> align with maxPlace

subset = X_test.loc[~fullgroup]

gap = 1.0 / (subset['maxPlace'].values - 1)

new_perc = np.around(subset['winPlacePerc'].values / gap) * gap  # half&up

X_test.loc[~fullgroup, 'winPlacePerc'] = new_perc



X_test['winPlacePerc'] = X_test['winPlacePerc'].clip(lower=0,upper=1)
# edge cases

X_test.loc[X_test['maxPlace'] == 0, 'winPlacePerc'] = 0

X_test.loc[X_test['maxPlace'] == 1, 'winPlacePerc'] = 1  # nothing

X_test.loc[(X_test['maxPlace'] > 1) & (X_test['numGroups'] == 1), 'winPlacePerc'] = 0

X_test['winPlacePerc'].describe()
test = pd.read_csv('../input/test_V2.csv')

test['matchId'] = test['matchId'].apply(lambda x: int(x,16))

test['groupId'] = test['groupId'].apply(lambda x: int(x,16))



submission = pd.merge(test, X_test[['matchId','groupId','winPlacePerc']])

submission = submission[['Id','winPlacePerc']]

submission.to_csv("submission.csv", index=False)
# from sklearn import linear_model

# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor

# from sklearn.tree import DecisionTreeRegressor

# from sklearn.metrics import mean_absolute_error, fbeta_score, make_scorer

# from sklearn.model_selection import GridSearchCV

# from sklearn.feature_selection import SelectFromModel

# import lightgbm as lgb

# from time import time

# from tqdm import tqdm
# X = train_data.drop(columns=['Id', 'groupId', 'matchId', 'winPlacePerc'])

# y = train_data['winPlacePerc']
# clf = RandomForestRegressor(n_estimators=10,random_state=0, criterion='mse')

# clf.fit(X, y)



# for feature in zip(X.columns, clf.feature_importances_):

#     print(feature)
# sfm = SelectFromModel(clf, threshold=0.004)

# sfm.fit(X, y)
# # Print the names of the most important features

# for feature_list_index in sfm.get_support(indices=True):

#     print(X.columns[feature_list_index])
# X_important = sfm.transform(X)
# def split_train_validation(data, n=100000):

#     return data[n:], data[:n]
# X_train, X_val = split_train_validation(X_important)

# y_train, y_val = split_train_validation(y)



# print('Training samples: {}'.format(len(X_train)))

# print('Validation samples: {}'.format(len(X_val)))
# X_train.shape
# data = lgb.Dataset(X_train, y_train)

# val_data = lgb.Dataset(X_val, y_val)
# param = {"objective" : "regression", "metric" : "mae", 'n_estimators':20000, 'early_stopping_rounds':200,

#               "num_leaves" : 31, "learning_rate" : 0.05, "bagging_fraction" : 0.7,

#                "bagging_seed" : 0, "num_threads" : 4,"colsample_bytree" : 0.7

#              }
# bst = lgb.train(param, data, valid_sets=[val_data], early_stopping_rounds=200, verbose_eval=1000)
# bst.save_model('model.txt', num_iteration=bst.best_iteration)
# print('Training Mean Absolute Error: ', mean_absolute_error(y_train, bst.predict(X_train)))
# def train_predict(learner, sample_size, X_train, y_train, X_val, y_val):

#     '''

#     inputs:

#         - learner: the learning algorithm to be trained and predicted on

#         - sample_size: the size of samples (number) to be drawn from training set

#         - X_train: features training set

#         - y_train: income training set

#         - X_val: features testing set

#         - y_val: income testing set

#     '''

#     results = {}

    

#     start = time()

#     learner = learner.fit(X_train[:sample_size], y_train[:sample_size])

#     end = time()

    

#     results['training_time'] = end - start



#     start = time()

#     predictions_val = learner.predict(X_val)

#     predictions_training = learner.predict(X_train)

#     end = time()

    

#     results['pred_time'] = end - start

    

#     results['mae_training'] = mean_absolute_error(y_train, predictions_training)

#     results['mae_val'] = mean_absolute_error(y_val, predictions_val)

    

#     print('{}: {} samples'.format(learner.__class__.__name__, sample_size))

#     print('Training MAE: {}'.format(results['mae_training']))

#     print('Val MAE: {}'.format(results['mae_val']))

#     print('Training time: {}'.format(results['training_time']))

#     print('Predictions time: {} \n\n'.format(results['pred_time']))
# model_A = RandomForestRegressor(n_estimators=10, max_depth=5, min_samples_leaf=5)

# model_B = GradientBoostingRegressor(n_estimators=10, max_depth=5, min_samples_leaf=5)

# model_C = BaggingRegressor(n_estimators=10)

# model_D = ExtraTreesRegressor(n_estimators=10, max_depth=5, min_samples_leaf=5)



# samples_100 = len(train_data)

# samples_10 = int(0.1 * samples_100)

# samples_1 = int(0.01 * samples_100)



# results = {}

# for model in [model_A, model_B, model_C, model_D]:

#     for i, samples in enumerate([samples_1, samples_10, samples_100]):

#         train_predict(model, samples, X_train, y_train, X_val, y_val)
# random_state = 42

# model = BaggingRegressor(random_state=random_state)



# parameters = {

#     'n_estimators': [10, 50, 100]

# }



# grid_scorer = make_scorer(mean_absolute_error)

# grid_obj = GridSearchCV(model, parameters, grid_scorer)

# grid_fit = grid_obj.fit(X_train, y_train)



# best_model = grid_fit.best_estimator_
# model = BaggingRegressor(base_estimator=None, bootstrap=True,

#          bootstrap_features=False, max_features=1.0, max_samples=1.0,

#          n_estimators=10, n_jobs=None, oob_score=False, random_state=42,

#          verbose=0, warm_start=False)



# model.fit(X_train, y_train)
# print('Training Mean Absolute Error: ', mean_absolute_error(y_train, model.predict(X_train)))
# print('Validation Mean Absolute Error: ', mean_absolute_error(y_val, model.predict(X_val)))
# test_data = pd.read_csv('test_V2.csv')

# test_data.head(5)
# train_data.drop(train_data[train_data.isnull().values == True].index, axis=0, inplace=True)



# reduce_mem_usage(train_data)



# train_data['totalDistance'] = train_data['walkDistance'] + train_data['rideDistance'] + train_data['swimDistance']

# train_data['damageDealtAndWalkDistance'] = train_data['damageDealt'] + train_data['walkDistance']

# train_data['items'] = train_data['boosts'] + train_data['heals']

# train_data['headshotRate'] = train_data['headshotKills'] / train_data['kills']

# train_data['headshotRate'].fillna(0, inplace=True)

# train_data['revivesHeals'] = train_data['revives'] * train_data['heals']



# train_data['numPlayers'] = train_data.groupby('matchId')['matchId'].transform('count')



# train_data['killsNorm'] = train_data['kills'] * ((100 - train_data['numPlayers']) / 100 + 1)

# train_data['damageDealt'] = train_data['damageDealt'] * ((100 - train_data['numPlayers']) / 100 + 1)

# train_data['maxPlace'] = train_data['maxPlace'] * ((100 - train_data['numPlayers']) / 100 + 1)

# train_data['matchDuration'] = train_data['matchDuration'] * ((100 - train_data['numPlayers']) / 100 + 1)



# train_data['killsWithoutMoving'] = ((train_data['kills'] > 0) & (train_data['totalDistance'] == 0))

# train_data.drop('killsWithoutMoving', axis=1, inplace=True)



# train_data.drop(train_data.query('Id == "c3e444f7d1289f"').index, inplace=True)



# query = '(weaponsAcquired > 50) & (totalDistance < 500)'

# train_data.drop(train_data.query(query).index, inplace=True)



# query = '(weaponsAcquired > 1) & (totalDistance == 0)'

# train_data.drop(train_data.query(query).index, inplace=True)



# query = '(weaponsAcquired > 100)'

# train_data.drop(train_data.query('Id == "da68d2812229a8"').index, inplace=True)



# query = 'walkDistance > 15000'

# train_data.drop(train_data.query(query).index, inplace=True)



# query = '(rideDistance > 30000)'

# train_data.drop(train_data.query(query).index, inplace=True)



# query = '(rideDistance == totalDistance) & (weaponsAcquired >= 1)'

# train_data.drop(train_data.query(query).index, inplace=True)



# query = 'swimDistance > 3000'

# train_data.drop(train_data.query(query).index, inplace=True)



# query = '(swimDistance == totalDistance) & (weaponsAcquired >= 1)'

# train_data.drop(train_data.query(query).index, inplace=True)



# query = 'heals > 50'

# train_data.drop(train_data.query(query).index, inplace=True)



# mType = ['solo', 'duo', 'squad', 'solo-fpp', 'duo-fpp', 'squad-fpp']

# idx = train_data[~train_data['matchType'].isin(mType)].index

# train_data.drop(idx, inplace=True)



# train_data = pd.get_dummies(train_data, columns=['matchType'])