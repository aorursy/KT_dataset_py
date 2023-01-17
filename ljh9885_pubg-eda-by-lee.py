import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

import matplotlib

matplotlib.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')

from sklearn.preprocessing import scale, minmax_scale

import os

import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error as mae

from sklearn.preprocessing import OneHotEncoder

import gc
def print_expand():

    pd.set_option('display.max_columns', None)  # or 1000

    pd.set_option('display.max_rows', 100)  # or 1000

    pd.set_option('display.max_colwidth', -1)  # or 199

def print_basic():

    pd.set_option('display.max_columns', 30)  # or 1000

    pd.set_option('display.max_rows', 30)  # or 1000

    pd.set_option('display.max_colwidth', 50)  # or 199

print_expand()
os.listdir('../input/pubg-finish-placement-prediction')
train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

sub = pd.read_csv('../input/pubg-finish-placement-prediction/sample_submission_V2.csv')
# def reduce_mem_usage(df, verbose=True):

#     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

#     start_mem = df.memory_usage().sum() / 1024**2    

#     for col in df.columns:

#         col_type = df[col].dtypes

#         if col_type in numerics:

#             c_min = df[col].min()

#             c_max = df[col].max()

#             if str(col_type)[:3] == 'int':

#                 if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

#                     df[col] = df[col].astype(np.int8)

#                 elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

#                     df[col] = df[col].astype(np.int16)

#                 elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

#                     df[col] = df[col].astype(np.int32)

#                 elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

#                     df[col] = df[col].astype(np.int64)  

#             else:

#                 if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

#                     df[col] = df[col].astype(np.float16)

#                 elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

#                     df[col] = df[col].astype(np.float32)

#                 else:

#                     df[col] = df[col].astype(np.float64)    

#     end_mem = df.memory_usage().sum() / 1024**2

#     if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

#     return df
# train = reduce_mem_usage(train)

# test = reduce_mem_usage(test)
print('train shape :', train.shape)

train.head()
print('test shape :', test.shape)

test.head()
print('sample_submission shape :', sub.shape)

sub.head()
train.isna().sum()
test.isna().sum()
train[train['winPlacePerc'].isna()==True]

# must be delete
train = train.dropna()

train.shape
train.describe()
train.describe(include=['O'])
train[train['winPlacePerc'] == 1].shape
train[train['matchType']=='solo-fpp']['DBNOs'].max()
train[train['matchId'] == "b30f3d87189aa6"].head()
d = train['groupId'].value_counts()

d = d[d > 4]

print('length :', len(d), 'max :', np.max(d), 'min :', np.min(d))

d[:5]
train[train['groupId'] == '23b79fb17aeaad']
def quan(x):

    print('quantile of {}'.format(x.name))

    for i in [i/10 for i in range(0, 10)]:

        print(f'{i} :', x.quantile(i))

    print('0.99 :', x.quantile(0.99))

    print('1.0 :', x.max())
train.columns
cols = train.columns[3:]

cols = cols.delete(12)

for col in cols:

    quan(train[col])

# maxPlace는 최대등수(꼴등)이지만 numGroup과 다를 수 있다.
def plot0(df, col, p='dist', c=1):

    if p == 'dist':

        if c == 0:

            fig, ax = plt.subplots(1, 2, figsize=(16, 7))

            sns.distplot(df[col], kde=False, ax=ax[0]).set(title = f'{col} dist.')

            sns.distplot(df[df[col] > 0][col], kde=False, ax=ax[1]).set(title = f'{col} > 0 dist.')

        elif c == 1:

            plt.figure(figsize=(15, 7))

            sns.distplot(df[col], kde=False).set(title = f'{col} dist.')

        else:

            print('c, E R R O R !')

    elif p == 'count':

        if c == 0:

            fig, ax = plt.subplots(1, 2, figsize=(16, 7))

            sns.countplot(df[col], ax=ax[0]).set(title = f'{col} dist.')

            sns.countplot(df[df[col] > 0][col], ax=ax[1]).set(title = f'{col} > 0 dist.')

        elif c == 1:

            plt.figure(figsize=(15, 7))

            sns.countplot(df[col]).set(title = f'{col} dist.')

        else:

            print('c, E R R O R !')

    else:

        print('p, E R R O R !')

        

def bplot(df, col1, col2):

    plt.figure(figsize=(15, 7))

    sns.boxplot(col1, col2, data=df).set(title = f'{col2} boxplot by {col1}')
plot0(train, 'winPlacePerc')
plot0(train, 'maxPlace', p='count')
plt.figure(figsize=(15, 7))

sns.countplot(train[((train['maxPlace'] >=20) & (train['maxPlace'] <= 30)) | ((train['maxPlace'] >= 45) & (train['maxPlace'] <= 55)) | ((train['maxPlace'] >= 90))]['maxPlace']).set(title = 'maxPlace range cut dist.')
plot0(train, 'numGroups', p='count')
train[train['maxPlace'] == 100].shape
train[train['numGroups'] == 100].shape
train[(train['maxPlace'] == 100) & (train['numGroups'] == 100)].shape
7 + 33 + 35 + 28 + 17 + 6 + 2 + 2 + 1

# 항상 maxPlace가 numGroups보다 크거나 같은듯
train[train['numGroups'] == 100]['matchId'].nunique()
train[train['maxPlace'] == 100]['matchId'].nunique()
tt = train[train['matchType'].str.contains('solo') == False]

print('not solo :', tt.shape)

tt.head()
plot0(tt, 'assists', p='count', c=0)
bplot(tt, 'assists', 'winPlacePerc')
plot0(tt, 'DBNOs', p='count', c=0)
bplot(tt, 'DBNOs', 'winPlacePerc')
plot0(tt, 'revives', p='count', c=0)
bplot(tt, 'revives', 'winPlacePerc')
del tt

gc.collect()
plot0(train, 'boosts', p='count', c=0)
bplot(train, 'boosts', 'winPlacePerc')
plot0(train, 'heals', p='count', c=0)
bplot(train, 'heals', 'winPlacePerc')
plot0(train, 'damageDealt', c=0)
plot0(train, 'kills', p='count', c=0)
plot0(train, 'kills', p='count')
bplot(train, 'kills', 'winPlacePerc')
plot0(train, 'killPlace', p='count')
bplot(train, 'killPlace', 'winPlacePerc')
plot0(train, 'killStreaks', p='count', c=0)
bplot(train, 'killStreaks', 'winPlacePerc')
train['killStreaks'].value_counts()
test['killStreaks'].value_counts()
plot0(train, 'headshotKills', p='count', c=0)
bplot(train, 'headshotKills', 'winPlacePerc')
plot0(train, 'longestKill', c=0)
plot0(train, 'roadKills', p='count', c=0)
plot0(train, 'teamKills', p='count', c=0)
plot0(train, 'matchDuration')
plt.figure(figsize=(15, 7))

sns.distplot(train[(train['matchDuration'] >= 1100) & (train['matchDuration'] <= 2200)]['matchDuration'], kde=False).set(title = 'matchDuration range cut dist.')
plt.figure(figsize=(15, 7))

sns.distplot(test['matchDuration'], kde=False)
train['matchDuration'].mean()
train['matchType'].value_counts()
plot0(train, 'killPoints', c=0)
plot0(train, 'rankPoints', c=0)
train[train['rankPoints'] == 0].shape
plot0(train, 'winPoints', c=0)
fig, ax = plt.subplots(1, 3, figsize=(16, 6))

sns.scatterplot('killPoints', 'winPlacePerc', data = train, ax=ax[0]).set(title = 'killPoints & winPlacePerc')

sns.scatterplot('rankPoints', 'winPlacePerc', data = train, ax=ax[1]).set(title = 'rankPoints & winPlacePerc')

sns.scatterplot('winPoints', 'winPlacePerc', data = train, ax=ax[2]).set(title = 'winPoints & winPlacePerc')
plot0(train, 'rideDistance', c=0)
plot0(train, 'swimDistance', c=0)
plot0(train, 'walkDistance')
plt.figure(figsize=(15, 7))

sns.scatterplot('walkDistance', 'winPlacePerc', data=train)
plt.figure(figsize=(15, 7))

sns.scatterplot('walkDistance', 'rideDistance', data=train)
plot0(train, 'vehicleDestroys', p='count', c=0)
plt.figure(figsize=(15, 7))

sns.boxplot('vehicleDestroys', 'winPlacePerc', data = train)
plt.figure(figsize=(14, 12))

sns.heatmap(train.iloc[:, 3:].corr(), annot=True, fmt='.2f')
train.columns
train['matchType'].value_counts()
# 실패작,,

# train.query("'squad' in matchType").shape



# tt = train[train['matchType'].str.contains('squad')]

# tt.head()



# squad4 = (tt.groupby('groupId')['Id'].count()[tt.groupby('groupId')['Id'].count() == 4]).index

# # size() 쓰면 됨



# train4 = tt[tt['groupId'].isin(squad4)]

# print(train4.shape)

# train4.head()



# train4[train4['winPlacePerc'] == 1].shape



# plt.figure(figsize = (15, 7))

# sns.distplot(train4['winPlacePerc'])



# train['winPlacePerc'].nunique()



# train4[train4['groupId'] == '4d4b580de459be']



# # train[train['matchId'] == 'a10357fd1a4a91'].sort_values('groupId')



# # train[train['groupId'] == '128b07271aa012']



# plt.figure(figsize = (15, 10))

# sns.heatmap(train.drop(['Id', 'groupId', 'matchId'], axis = 1).corr(), annot = True, fmt = '.2f')



# gb = train4.groupby('groupId')



# kill = gb['kills'].std()

# win = gb['winPlacePerc'].mean()



# walk = gb['walkDistance'].std()

# d = walk.reset_index()

# tr = train4.copy()

# d = pd.merge(tr, walk, on = 'groupId', how = 'left')

# d.head()



# np.sum(d['walkDistance_x'] < d['walkDistance_y'])

# # 원데이터보다 표준편차가 더 작은 경향



# def csplot(col, sv = 'std'):

#     if sv == 'std':

#         c = gb[col].std()

#         win = gb['winPlacePerc'].mean()

#         df = pd.concat([c, win], axis = 1)

#         fig, ax = plt.subplots(2, 2, figsize = (15, 10))

#         sns.heatmap(df.corr(), annot = True, fmt = '.2f', ax = ax[0, 0]).set(title = f"Groupby groupId({sv}) corr")

#         sns.heatmap(train4[[col, 'winPlacePerc']].corr(), annot = True, fmt = '.2f', ax = ax[0, 1]).set(title = "All data corr")

#         sns.scatterplot(x = 'winPlacePerc', y = col, data = df, ax = ax[1, 0]).set(title = f"Groupby groupId({sv}) plot")

#         sns.scatterplot(x = 'winPlacePerc', y = col, data = train4, ax = ax[1, 1]).set(title = "All data plot")

#         plt.tight_layout()

#         plt.show()

#     elif sv == 'var':

#         c = gb[col].std() ** 2

#         win = gb['winPlacePerc'].mean()

#         df = pd.concat([c, win], axis = 1)

#         fig, ax = plt.subplots(2, 2, figsize = (15, 10))

#         sns.heatmap(df.corr(), annot = True, fmt = '.2f', ax = ax[0, 0]).set(title = f"Groupby groupId({sv}) corr")

#         sns.heatmap(train4[[col, 'winPlacePerc']].corr(), annot = True, fmt = '.2f', ax = ax[0, 1]).set(title = "All data corr")

#         sns.scatterplot(x = 'winPlacePerc', y = col, data = df, ax = ax[1, 0]).set(title = f"Groupby groupId({sv}) plot")

#         sns.scatterplot(x = 'winPlacePerc', y = col, data = train4, ax = ax[1, 1]).set(title = "All data plot")

#         plt.tight_layout()

#         plt.show()

#     else:

#         print("error")



# csplot('kills')



# csplot('kills', 'var')



# csplot('damageDealt')



# csplot('damageDealt', 'var')



# csplot('boosts')



# csplot('boosts', 'var')



# csplot('walkDistance')



# csplot('walkDistance', 'var')
plt.figure(figsize=(15, 7))

sns.countplot(train['maxPlace'])
np.sum(test['maxPlace'] == 2)
# Feature Engineering
# 탈것을 파괴 했다 안했다로 바꾸기

# 차를 탔다 안탔다 바꾸기

# 이동 수단 상관없이 거리 다 합치기

# ELO는 매치별로 minmax 하기

# killPlace는 매치별로 minmax 하기

# matchType은 squad, squad-fpp, duo, duo-fpp, solo, solo-fpp, 그 외 나머지

# killStreaks는 4이상 통합하기

# DBNOs는 6이상 통합하기

# headshotKills는 4이상 통합하기

# kills는 8이상 통합하기

# 그룹별 변수가 뭔가 중요한 것 같은 느낌 -> 아예 그룹의 합킬, 합딜, 평균딜, 평균킬 같은 거?????

# 개인보다 팀성과였음 (같은 그룹이면 등수가 같기 때문)
headshot = train[['kills', 'headshotKills', 'winPlacePerc']]

headshot['headshotrate'] = 0

headshot['headshotrate'][headshot['kills'] > 0] = headshot['kills'] / headshot['headshotKills']

headshot.corr()
# del train, test

# gc.collect()
# def data_transform(df, is_train=True):

# #     if is_train:

# #         df = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')

# #         df = df.dropna()

# #     else:

# #         df = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')

        

#     print('make features..')

#     df['headshotRate'] = 0

#     df['headshotRate'][df['kills'] > 0] = df['headshotKills'] / df['kills']

#     df['killStreaksPerc'] = 0

#     df['killStreaksPerc'][df['kills'] > 0] = df['killStreaks'] / df['kills']

#     df['roadKillPerc'] = 0

#     df['roadKillPerc'][df['kills'] > 0] = df['roadKills'] / df['kills']

#     df['matchDuration'] = df['matchDuration'].map(lambda x: 0 if x < 1600 else 1)

#     for c in ['vehicleDestroys', 'teamKills']:

#         df[c] = df[c].map(lambda x: 1 if x > 0 else 0)

    

#     l1 = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals',

#           'kills', 'killStreaks', 'longestKill', 'revives', 'rideDistance', 'roadKills',

#           'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',

#           'headshotRate', 'killStreaksPerc', 'roadKillPerc']

#     l2 = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals',

#           'kills', 'killStreaks', 'longestKill', 'revives', 'rideDistance', 'roadKills',

#           'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired',

#           'headshotRate', 'killStreaksPerc', 'roadKillPerc', 'killPlacePerc', 'Points']

#     l3 = [['squad-fpp', 'normal-squad-fpp', 'squad', 'normal-squad'],

#           ['duo', 'duo-fpp', 'normal-duo-fpp', 'normal-duo'],

#           ['solo-fpp', 'solo', 'normal-solo-fpp', 'normal-solo'],

#           ['crashfpp', 'flaretpp', 'flarefpp', 'crashtpp']]

#     dic1 = {}

#     for i in range(len(l3)):

#         for j in range(len(l3[i])):

#             dic1[l3[i][j]] = i

#     df['matchType'] = df['matchType'].map(lambda x: dic1[x])

    

#     gb_mg = df.groupby(['matchId', 'groupId'])

#     gb_m = df.groupby('matchId')

    

#     if is_train:

#         y_train = gb_mg['winPlacePerc'].mean().values

#     else:

#         y_train = None

    

#     df['killPlacePerc'] = gb_m['killPlace'].rank(pct=True, ascending=False).values

#     df['Points'] = df['killPoints'] + df['rankPoints'] + df['winPoints']

#     df['Points'] = gb_m['Points'].rank(pct=True, ascending=False).values

#     df_n = gb_mg.size().reset_index().rename(columns = {0:'n_team'})

#     df = df.merge(df_n, how='left', on=['matchId', 'groupId'])

    

#     print('make group mean features')

#     agg = gb_mg[l1].agg('mean')

#     agg_rank = agg.groupby('matchId')[l1].rank(pct=True).reset_index()

    

#     if is_train == True: output = agg.reset_index()[['matchId', 'groupId']]

#     else: output = df[['matchId', 'groupId']]

    

#     output = output.merge(agg.reset_index(), how='left', on=['matchId', 'groupId'])

#     output = output.merge(agg_rank, suffixes=["_mean", "_mean_rank"], how='left', on=['matchId', 'groupId'])

    

#     print('make group sum features')

#     agg = gb_mg[l1].agg('sum')

#     agg_rank = agg.groupby('matchId')[l1].rank(pct=True).reset_index()

#     output = output.merge(agg.reset_index(), how='left', on=['matchId', 'groupId'])

#     output = output.merge(agg_rank, suffixes=["_sum", "_sum_rank"], how='left', on=['matchId', 'groupId'])

    

#     print('make group min features')

#     agg = gb_mg[l2].agg('min')

#     agg_rank = agg.groupby('matchId')[l2].rank(pct=True).reset_index()

#     output = output.merge(agg.reset_index(), how='left', on=['matchId', 'groupId'])

#     output = output.merge(agg_rank, suffixes=["_min", "_min_rank"], how='left', on=['matchId', 'groupId'])

    

#     print('make group max features')

#     agg = gb_mg[l2].agg('max')

#     agg_rank = agg.groupby('matchId')[l2].rank(pct=True).reset_index()

#     output = output.merge(agg.reset_index(), how='left', on=['matchId', 'groupId'])

#     output = output.merge(agg_rank, suffixes=["_max", "_max_rank"], how='left', on=['matchId', 'groupId'])

    

#     print('matchType transform & make n_team & maxPlace & matchDuration')

#     output = output.merge(gb_mg.size().reset_index(name='n_team'), how='left', on=['matchId', 'groupId'])

#     output = output.merge(gb_m[['matchType', 'maxPlace', 'matchDuration']].agg('mean').reset_index(), how='left', on='matchId')

    

#     print('make match mean features')

#     agg = gb_m[l1].agg('mean').reset_index()

#     output = output.merge(agg, suffixes=["", "_match_mean"], how='left', on=['matchId'])

    

#     print('make match max features')

#     agg = gb_m[l1].agg('max').reset_index()

#     output = output.merge(agg, suffixes=["", "_match_max"], how='left', on=['matchId'])

    

#     output.drop(["matchId", "groupId"], axis=1, inplace=True)



#     del df, df_n, l1, l2, l3, gb_mg, gb_m

#     gc.collect()

#     print('complete!')

    

#     return output, y_train
# %%time

# x_train, y_train = data_transform(train)

# x_test, _ = data_transform(test, False)
# print('transformed train shape :', x_train.shape)

# x_train.head()
# print('transformed test shape :', x_test.shape)

# x_test.head()
# print('y_train length', len(y_train))

# y_train[:5]
# ohe = OneHotEncoder(handle_unknown = 'ignore')

# ohe_train = pd.DataFrame(ohe.fit_transform(train[['matchType']]).toarray())

# ohe_test = pd.DataFrame(ohe.transform(test[['matchType']]).toarray())

# ohe_train.head()
# ohe_train = pd.DataFrame(ohe.fit_transform(train[['matchType']]).toarray())

# ohe_test = pd.DataFrame(ohe.transform(test[['matchType']]).toarray())
# train.to_csv('train_preprocessing.csv', index=False)

# test.to_csv('test_preprocessing.csv', index=False)
# Modeling
# x_train = train.drop(['Id', 'groupId', 'matchId', 'winPlacePerc'], axis = 1)

# y_train = train['winPlacePerc']

# x_test = test.drop(['Id', 'groupId', 'matchId'], axis = 1)

# x_train.head()
# xgb_params={'eta':0.1,

#             'max_depth':6,

#             'objective':'reg:squarederror',

#             'eval_metric':'mae',

#             'seed':74,

#             'tree_method':'gpu_hist',

#             'predictor':'gpu_predictor'}
# %%time



# kf = KFold(n_splits=3, shuffle=True, random_state=42)

# oof = np.zeros(len(train))

# predictions = np.zeros(len(test))



# for fold_, (trn_idx, val_idx) in enumerate(kf.split(x_train, y_train)):

#     print("fold num_: {}".format(fold_))

#     trn_data = xgb.DMatrix(x_train.iloc[trn_idx], label=y_train.iloc[trn_idx])

#     val_data = xgb.DMatrix(x_train.iloc[val_idx], label=y_train.iloc[val_idx])

    

#     watchlist = [(trn_data, 'train'), (val_data, 'valid')]

#     num_round = 10000

#     model = xgb.train(params = xgb_params,

#                       dtrain = trn_data,

#                       num_boost_round  = num_round,

#                       evals = watchlist,

#                       verbose_eval = 100,

#                       early_stopping_rounds = 50

#                 )

#     oof[val_idx] = model.predict(xgb.DMatrix(x_train.iloc[val_idx]), ntree_limit = model.best_iteration)



    

#     predictions += model.predict(xgb.DMatrix(x_test), ntree_limit = model.best_iteration) / 5

    

# print('\nCross Validation Is Complete')

# print("CV score: {:<8.5f}".format(mae(y_train, oof)))
# fig,ax = plt.subplots(figsize = (10,10))

# xgb.plot_importance(model, ax = ax)

# plt.show()
# param = {'num_leaves': 31,

#          'learning_rate': 0.1,

#          'max_depth': 7,

#          'seed': 2020,

#          'objective': 'regression',

#          'boosting_type': 'gbdt',

#          'metric': 'mae'}
# %%time



# kf = KFold(n_splits=3, shuffle=True, random_state=42)

# oof_lgb = np.zeros((len(train)))

# lgb_pred = np.zeros((len(test)))



# for fold_, (trn_idx, val_idx) in enumerate(kf.split(x_train, y_train)):

#     print("fold num_: {}".format(fold_))

#     trn_data = lgb.Dataset(x_train.iloc[trn_idx], label=y_train[trn_idx])

#     val_data = lgb.Dataset(x_train.iloc[val_idx], label=y_train[val_idx])

    

#     num_round = 15000

#     model = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 50)

#     oof_lgb[val_idx] = model.predict(x_train.iloc[val_idx], num_iteration=model.best_iteration)    

#     lgb_pred += model.predict(x_test, num_iteration=model.best_iteration) / 5

    

# print('\nCross Validation Is Complete')

# print("CV score: {:<8.5f}".format(mae(y_train, oof_lgb)))
# fig, ax = plt.subplots(figsize=(10, 10))

# lgb.plot_importance(model, ax=ax, max_num_features=50)

# plt.show()
# sub['winPlacePerc'] = lgb_pred

# sub.head()
# sub['winPlacePerc'] = sub['winPlacePerc'].map(lambda x: 1 if x >= 1 else x)

# sub['winPlacePerc'] = sub['winPlacePerc'].map(lambda x: 0 if x <= 0 else x)

# sub.head()
# sub.to_csv('submission.csv', index = False)