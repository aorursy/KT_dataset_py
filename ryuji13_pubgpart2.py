#標準的なライブラリ

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))



#可視化のライブラリ

import matplotlib.pyplot as plt

import seaborn as sns

from pdpbox import pdp

from plotnine import *

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor

from IPython.display import display



#機械学習

import sklearn

from scipy.cluster import hierarchy as hc

from fastai.imports import *



#

%load_ext autoreload

%autoreload 2

%matplotlib inline

train = pd.read_csv('../input/train_V2.csv')

test = pd.read_csv('../input/test_V2.csv')
display(train.head())



display(train.tail())
train.describe()
train.info()



print('the shape is : ', train.shape)
train[train['winPlacePerc'].isnull()]
#削除

train.drop(2744604, inplace=True)
train[train['winPlacePerc'].isnull()]
#プレイヤー数

train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

plt.figure(figsize=(15,10))

sns.countplot(train[train['playersJoined']>=75]['playersJoined'])

plt.title('プレイヤー数')

plt.show()
#既存の関数で標準化するのかと思ったら違った

train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 +1)

train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 +1)

train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 +1)

train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 +1)



#出力して確認

to_show = ['Id', 'kills', 'killsNorm', 'damageDealt', 'damageDealtNorm', 'maxPlace',

           'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']

train[to_show][0:20]
train['healsandboosts'] = train['heals'] + train['boosts']

to_show = ['heals', 'boosts', 'healsandboosts']

train[to_show].tail()
#総移動距離

train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']



#動かずに倒したフラグ

train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
train['killsWithoutMoving'].value_counts()
#削除

train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
train[train['roadKills'] > 10]#.shape
plt.figure(figsize=(12, 4))

sns.countplot(data=train, x=train['kills']).set_title('Kills')

plt.show()
display(train[train['kills'] > 30].shape)

train[train['kills'] > 30].head(10)
train.drop(train[train['kills'] > 30].index, inplace=True)
plt.figure(figsize=(12, 4))

sns.distplot(train['headshotKills'], bins=10)

plt.show()
plt.figure(figsize=(12, 4))

sns.distplot(train['longestKill'], bins=10)

plt.show()
display(train[train['longestKill'] > 1000].shape)

train[train['longestKill'] >= 1000].head()
train[['walkDistance', 'rideDistance', 'swimDistance', 'totalDistance']].describe()
plt.figure(figsize=(12, 4))

sns.distplot(train['walkDistance'], bins=10)

plt.show()
display(train[train['walkDistance'] >= 10000].shape)

train[train['walkDistance'] > 10000].head()
plt.figure(figsize=(12, 4))

sns.distplot(train['swimDistance'], bins=10)

plt.show()
# Players who swam more than 2 km

train[train['swimDistance'] >= 2000]
train['matchType'].nunique()
#ワンホットエンコード

train = pd.get_dummies(train, columns=['matchType'])



#正規表現で抜き出し

matchType_encoding = train.filter(regex='matchType')



matchType_encoding.head()
#groupIdとmatchIdをカテゴリー値に変換



train['groupId'] = train['groupId'].astype('category')

train['matchId'] = train['matchId'].astype('category')



train['groupId_cat'] = train['groupId'].cat.codes

train['matchId_cat'] = train['matchId'].cat.codes



train.drop(columns=['groupId', 'matchId'], inplace=True)



train[['groupId_cat', 'matchId_cat']].head()
sample = 500000

df_sample = train.sample(sample)
df = df_sample.drop(columns = ['winPlacePerc'])

y = df_sample['winPlacePerc']
def split_vals(a, n : int):

    return a[:n].copy(), a[n:].copy()



val_perc = 0.12 #検証に使う分

n_valid = int(val_perc * sample)

n_trn = len(df)-n_valid



raw_train, raw_valid = split_vals(df_sample, n_trn)

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



print('Sample train shape: ', X_train.shape,

      'Sample target shape: ', y_train.shape,

      'Sample validation shape: ', X_valid.shape)
#MAE mean absolute error 平均絶対誤差

from sklearn.metrics import mean_absolute_error



def print_score(m : RandomForestRegressor):

    res = ['mae train: ', mean_absolute_error(m.predict(X_train2), y_train),

           'mae val: ', mean_absolute_error(m.predict(X_valid2), y_valid)]

    

    if hasattr(m, 'oob_score_'):res.append(m.oob_score_)

    print(res)
#IDが邪魔っぽい

X_train.columns

X_train['Id']

X_train2 = X_train.drop(['Id'], axis=1)

X_valid2 = X_valid.drop(['Id'], axis=1)

X_train2.shape

#y_train
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m1.fit(X_train2, y_train)

print_score(m1)
def rf_feat_importance(m, df):

    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}

                       ).sort_values('imp', ascending=False)



df = df.drop(['Id'], axis=1)

fi = rf_feat_importance(m1, df); fi[:10]
plot1 = fi[:20].plot('cols', 'imp', figsize=(14, 6), legend=False, kind='barh')

plot1
to_keep = fi[fi.imp>0.005].cols



#重要な特徴

print('Significant features: ', len(to_keep))

to_keep
#重要な特徴量だけのデータフレームを作る

df_keep = df[to_keep].copy()

y_keep = y[to_keep].copy()

X_train, X_valid = split_vals(df_keep, n_trn)

y_train, y_valid = split_vals(y_keep, n_trn)
X_train.shape



#X_train2 = X_train.drop(['Id'], axis=1)

#X_valid2 = X_valid.drop(['Id'], axis=1)
#m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m2 = RandomForestRegressor(n_estimators=80, min_samples_leaf=3, max_features='sqrt', n_jobs=-1)

m2.fit(X_train, y_train)

print_score(m2)
fi_to_keep = rf_feat_importance(m2, df_keep)

plot2 = fi_to_keep.plot('cols', 'imp', figsize=(14, 6), legend=False, kind='barh')

plot2