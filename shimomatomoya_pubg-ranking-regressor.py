# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import time

import lightgbm as lgb

from sklearn.model_selection import KFold, cross_validate, GridSearchCV

import math

import tqdm
#まずはデータをDF化

train_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

test_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')

sample_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv')
#表示カラム数を拡張

pd.set_option('display.max_columns', 40)
# 各カラムを確認していく

# Id プレイヤーのIDです。

# groupID 4人までのグループで振り分けられるID

# matchID 100人までの試合ごと振り分けられるID

# assists ダメージを与えたが、仲間が敵を倒した回数

# boosts エナドリ使用回数

# damagedealt 合計ダメージ

# DBNOs 敵からダウンをもらった回数

# headshotKills ヘッドショットで倒した回数

# heals ヒーリングアイテム使用回数(包帯)

# killPlace 敵を倒した数のランキング

# killpoints ダメージベースのランキング？ここちょっと不明わからず。

# kills 倒したプレイヤーの数

# killstreaks 連続して倒した数

# longestKill プレイヤーを倒した時の最長距離

# matchDuration マッチ時間(秒

# mathType ゲームタイプ

# maxPlace 最悪の順位（？）numGroupsと数字が合わないことがある

# numGroups グループの順位

# rankPoints -1,0の場合はkillPointsかwinPointsにポイントが入っている。

# revives 仲間を助けた回数

# rideDistance 車両で走行した総距離（メートル） 

# roadKills 車で敵を倒した数

# swimDistance 泳いだ距離(メートル)

# teamKills 仲間を倒してしまった数

# vehicleDestroys 車両を壊した数

# walkDistance 歩いた距離（メートル)

# weaponsAcquired 武器を拾った数

# winPoints 勝利点数(これも詳しくは不明)

# winPlacePerc 1になるほど結果勝ってた人
#トレーニングデータ確認

train_df
#欠損値確認

train_df.isnull().sum()
#winPlacePercにある欠損値の行を削除。

train_df.drop(train_df[train_df['winPlacePerc'].isnull()].index, axis=0, inplace=True)

train_df.reset_index(drop=True, inplace=True)
#あまり理解できない3種のポイントを調べる

pointcheck = train_df[['killPoints', 'rankPoints', 'winPoints']]
print( len(pointcheck))

print( len( pointcheck[pointcheck['rankPoints'] == -1]))

print( len( pointcheck[pointcheck['killPoints'] != 0]))

print( len( pointcheck[pointcheck['winPoints'] != 0]))
#rankPointsが-1 or 0 の人は基本的にkillPoints,winPointsが入ってる

print(pointcheck[pointcheck['rankPoints'] != -1][pointcheck[pointcheck['rankPoints'] != -1]['winPoints'] != 0])

print(pointcheck[pointcheck['rankPoints'].isin([0, -1])])
#kill,rank,win,winPlacePercについて見てみる

#散布図描画

fig, ax = plt.subplots(1, 3, figsize=(48,12))

ax[0].scatter(x = list(train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']].query('killPoints != 0')['killPoints']),

           y = list(train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']].query('killPoints != 0')['winPlacePerc']))

ax[0].set_xlabel('killPoints')

ax[0].set_ylabel('winPlacePerc')

ax[0].set_title('killPoints & winPlacePerc')



ax[1].scatter(x = list(train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']].query('winPoints != 0')['winPoints']),

           y = list(train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']].query('winPoints != 0')['winPlacePerc']))

ax[1].set_xlabel('winPoints')

ax[1].set_ylabel('winPlacePerc')

ax[1].set_title('winPoints & winPlacePerc')



ax[2].scatter(x = list(train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']][train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']]['rankPoints'].isin([-1, 0]) == False]['rankPoints']),

           y = list(train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']][train_df[['killPoints', 'rankPoints', 'winPoints','winPlacePerc']]['rankPoints'].isin([-1, 0]) == False]['winPlacePerc']))

ax[2].set_xlabel('rankPoints')

ax[2].set_ylabel('winPlacePerc')

ax[2].set_title('rankPoints & winPlacePerc')



plt.show()



#高いから勝ちやすい というわけでも無さそう

#カラムから削除する
#matchTypeはダミー変数にしたいので、要素数確認。

print(sorted(train_df['matchType'].unique()))

print(sorted(test_df['matchType'].unique()))
#何も動いていない & killがあるチーターの疑いがある人を消す

train_df['total_move'] = train_df['walkDistance'] + train_df['swimDistance'] + train_df['rideDistance']

train_df['not_move_playerkill'] = ((train_df['kills'] > 0) & (train_df['total_move'] == 0))

train_df.drop( train_df[train_df['not_move_playerkill'] == True].index, inplace=True)

train_df.reset_index(drop=True,inplace=True)
#量的変数 kill,rank,winPointsの3種を除く。

quantiles = list( train_df.columns[3:10] ) + list( train_df.columns[12:15] ) + list( train_df.columns[16:18] ) + list( train_df.columns[19:27])



#カテゴリ変数 matchType

categories = list(train_df[['matchType']].columns)



#目的変数 winPlacePerc

y = list(train_df[['winPlacePerc']].columns)
#カテゴリ変数をpd.getdummiesし、再度結合する。

train_df_1 = pd.merge(pd.merge(train_df[quantiles], pd.get_dummies(train_df[categories]), how='inner', left_index=True, right_index=True),

                    train_df[y], how='inner', left_index=True, right_index=True)

test_df_1 = pd.merge(test_df[quantiles], pd.get_dummies(test_df[categories]), how='inner', left_index=True, right_index=True)
#プロット時の初期設定

plt.style.use('seaborn-darkgrid')

plt.rcParams['font.family'] = 'Yu Gothic'

plt.rcParams['font.size'] = 10
#相関係数のヒートマップ

#どうやら歩く距離が長いほどランキングが上がるよう

#killPlaceとは負の相関がある

#武器をより拾う、エナドリをたくさん飲む人もランキングは上がる傾向にある

f,ax = plt.subplots(figsize=(15, 15))

sns.heatmap(train_df_1.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
#assistsはそこまでwinPlacePercに影響はしていなさそう

plt.figure(figsize=(15,15))

sns.jointplot(data=train_df_1, x='winPlacePerc', y='assists', height=10, ratio=3, color='royalblue')

plt.show()
#車両破壊台数とwinPlacePercの関係性

#0 or 1overで結構差がある

plt.figure(figsize=(15,15))

sns.pointplot(data=train_df_1, x='vehicleDestroys',y='winPlacePerc', color='royalblue',

             alpha=1)

plt.xlabel('車両破壊台数')

plt.ylabel('winPlacePerc')

plt.grid()

plt.show()
#healsとwinPlacePercの関係性

#ある程度相関はある

plt.figure(figsize=(15,15))

sns.jointplot(data=train_df_1, x='winPlacePerc', y='heals', height=10, ratio=3, color='royalblue')

plt.show()
#説明変数を格納

x = list(train_df_1.columns[:-1])

print(x)

#目的変数は元から変数に格納してる

print(y)
#最小二乗法重回帰で一旦係数を見てみる

import statsmodels.api as sm

import statsmodels.formula.api as smf

X = sm.add_constant(train_df_1[x])

model = sm.OLS(train_df_1[y], X)

result = model.fit()

result.summary()
#LightGBM用にデータを加工

x_data = train_df_1[x].values #説明変数

y_data = train_df_1[y].values.reshape((-1, 1)) #目的変数
import lightgbm as lgb
#kaggle karnelのメモリ不足のため、変数を一部消します。

del train_df
import gc

gc.collect()
#デフォルトパラメータで交差検証 n_splitsはこれ以上上げるとメモリ足りない

LGBMR = lgb.LGBMRegressor()

cv_split = KFold( n_splits= 3, random_state=1, shuffle=True)

base_results = cross_validate(LGBMR, x_data, y_data,

                             scoring=('r2', 'neg_mean_squared_error'), cv=cv_split, n_jobs=-1)



print(np.mean(base_results['test_r2']))

print(-np.mean(base_results['test_neg_mean_squared_error']))

#決定係数は0.92

#下は平均二乗誤差
%%time

#一番良かったパラメータでfitさせる

#ハイパーパラメータは時間かかるのでローカルのJupyter Notebookで実行しました。

LGBMR = lgb.LGBMRegressor(boosting_type='gbdt',

                         learning_rate=0.15,

                          max_depth=-1,

                          min_child_samples=10,

                          min_child_weight=0.001,

                          min_split_gain=0,

                          n_estimators=100,

                          n_jobs=-1,

                          num_leaves=128,

                          random_state=None,

                          reg_alpha=0,

                          reg_lambda=0.01,

                          silent=True,

                          subsample_for_bin=200000,

                          subsample_freq=0

                         )

LGBMR.fit(x_data, y_data)
#predictに入れるテストデータ

test_df_1[x].values
#予測

test_y = LGBMR.predict(test_df_1[x].values)

#結果

print(test_y)
#提出時ファイル確認

sample_df
#test_dfのIdと予測結果のwinPlacePercをくっつける

submission_df = pd.merge(test_df[['Id']], pd.DataFrame(test_y, columns=['winPlacePerc']), how='inner', left_index=True, right_index=True)

submission_df
submission_df.to_csv('submission.csv', index=False)