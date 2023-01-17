# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import matplotlib.pyplot as plt

%matplotlib inline
### Your code blocks here

# 分析・説明のため、まずはダメなモデル構築・評価を一通り実行.



import lightgbm as lgb

from sklearn import preprocessing

import matplotlib.pyplot as plt

%matplotlib inline



train_df = pd.read_csv("../input/ds2019uec-task1/train.csv")

test_df = pd.read_csv("../input/ds2019uec-task1/test.csv")



train = train_df.copy()

test = test_df.copy()



encoders = dict()

for col in ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"]:

    le = preprocessing.LabelEncoder()

    le.fit(pd.concat([train_df[col], test_df[col]], axis=0))

    train[col] = le.transform(train_df[col])

    test[col] = le.transform(test_df[col])

    encoders[col] = le

    

train["y"] = (train_df["y"] == "yes").astype(np.int)



train_data = lgb.Dataset(train.drop("y", axis=1), label=train["y"])



param = {'num_leaves': 31, 'objective': 'binary'}

param['verbose'] = -1

param['metric'] = 'auc'



cv_result = lgb.cv(param, train_data, 50, nfold=5, verbose_eval=False)



np.array(cv_result["auc-mean"]).argmax()



bst = lgb.train(param, train_data, 42)



ypred = bst.predict(test)



sub = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

sub["pred"] = ypred

sub.to_csv("sample_lgbm.csv", index=False)
#まずはデータの中身を見てみる train側

train_df.head()
#データの中身を見てみる test側

test_df.head()
# 欠損値と数値データの基礎統計量の確認.

print(train_df.info())

train_df.describe()
# 欠損値と数値データの基礎統計量の確認.

print(test_df.info())

test_df.describe()
# trainの月ごとの成約率を見てみる.5～4月の一年間のデータのため単純に月でグルーピング.

train_tmp = train_df

train_tmp['y'] = (train_df["y"] == "yes").astype(np.int)



# 順番に並べるため月を数字に変換、

train_mn = train_tmp.replace({'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12,'jan':13,'feb':14,'mar':14,'apr':16})



# 各月のレコード数

print("------------各月のレコード数------------") 

print(train_mn.groupby('month').size())



# 各月の成約率

print("------------各月の成約率------------") 

print(train_mn.groupby('month').mean()['y'])

#成約率をグラフ化

plt.figure();

y = train_mn.groupby('month').mean()['y'].plot(kind='bar'); plt.axhline(0, color='k')

plt.show()
lgb.plot_importance(bst, figsize=(12, 6))
train_y_pred = bst.predict(train.drop("y", axis=1))

plt.hist(train_y_pred)
plt.hist(ypred)
train_x = train.drop(['y','index', 'duration'], axis=1)

train_y = train["y"]



# 変数がちゃんと落ちているか

train_x.head()
# データの分割の為、indexの順にperiod変数を作成し4分割する



# 4分割した数で切り捨て除算することで、trainにそれぞれindexごと4分割した数を入れる.

train_x['period'] = np.arange(0, len(train)) // (len(train) // 4) 



# 0から3に収める.

train_x['period'] = np.clip(train_x['period'], 0, 3)



print(train_x.head())

print(train_x.tail())
# testはとりあえずperiodをすべて4とする.

test['period'] = 4
train_data = lgb.Dataset(train_x, label=train["y"])



param = {'num_leaves': 31, 'objective': 'binary'}

param['verbose'] = -1

param['metric'] = 'auc'
# 時系列考慮した交差検証法を行う.

va_period_list = [1, 2, 3]

for va_period in va_period_list:

    is_tr =train_x['period'] < va_period

    is_va=train_x['period'] == va_period

    tr_x, va_x = train_x[is_tr], train_x[is_va]

    tr_y, va_y = train_y[is_tr], train_y[is_va]

    

    train_data = lgb.Dataset(tr_x, tr_y)

    valid_data = lgb.Dataset(va_x, va_y)

    

    #cvでなく、データセットを指定して直接trainを呼ぶ

    lgb.train(param, train_data, 50, valid_sets=[train_data, valid_data], verbose_eval=True)    



#以上を、train0→train1 train0～1→train2　train0～2→train3　の予測を繰り返す
#全データを学習データに.

train_all_data = lgb.Dataset(train_x, train_y)



bst = lgb.train(param, train_all_data, 20, valid_sets=[train_all_data], verbose_eval=True)
#変数重要度をチェック



lgb.plot_importance(bst, figsize=(12, 6))
test_x = test.drop(["index", "duration"], axis=1)

y_pred = bst.predict(test_x)
sub = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

sub["pred"] = y_pred

sub.to_csv("sample_lgbm.csv", index=False)
train_y_pred = bst.predict(train_x)

plt.hist(train_y_pred)
plt.hist(y_pred)
#成約率をグラフ化

plt.figure();

y = train.groupby('day').mean()['y'].plot(kind='bar'); plt.axhline(0, color='k')

plt.show()
train_x = train.drop(['y','index', 'duration','day','month'], axis=1)

train_y = train["y"]



# データの分割の為、indexの順にperiod変数を作成し4分割する



# 4分割した数で切り捨て除算することで、trainにそれぞれindexごと4分割した数を入れる.

train_x['period'] = np.arange(0, len(train)) // (len(train) // 4) 



# 0から3に収める.

train_x['period'] = np.clip(train_x['period'], 0, 3)



# testはとりあえずperiodをすべて4とする.

test['period'] = 4



train_data = lgb.Dataset(train_x, label=train["y"])



param = {'num_leaves': 31, 'objective': 'binary'}

param['verbose'] = -1

param['metric'] = 'auc'



# 時系列考慮した交差検証法を行う.

va_period_list = [1, 2, 3]

for va_period in va_period_list:

    is_tr =train_x['period'] < va_period

    is_va=train_x['period'] == va_period

    tr_x, va_x = train_x[is_tr], train_x[is_va]

    tr_y, va_y = train_y[is_tr], train_y[is_va]

    

    train_data = lgb.Dataset(tr_x, tr_y)

    valid_data = lgb.Dataset(va_x, va_y)

    

    #cvでなく、データセットを指定して直接trainを呼ぶ

    lgb.train(param, train_data, 50, valid_sets=[train_data, valid_data], verbose_eval=True)    



#以上を、train0→train1 train0～1→train2　train0～2→train3　の予測を繰り返す
#全データを学習データに.

train_all_data = lgb.Dataset(train_x, train_y)



bst = lgb.train(param, train_all_data, 20, valid_sets=[train_all_data], verbose_eval=True)
#変数重要度をチェック



lgb.plot_importance(bst, figsize=(12, 6))
test_x = test.drop(["index", "duration","day","month"], axis=1)

y_pred = bst.predict(test_x)
sub = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

sub["pred"] = y_pred

sub.to_csv("sample_lgbm.csv", index=False)
train_y_pred = bst.predict(train_x)

plt.hist(train_y_pred)
plt.hist(y_pred)
#10 月のデータを削除

train_rej_oct = train[train['month'] != 'oct']

train_x = train_rej_oct.drop(['y','index', 'duration','day','month'], axis=1)

train_y = train_rej_oct["y"]

# データの分割の為、indexの順にperiod変数を作成し4分割する

# 4分割した数で切り捨て除算することで、trainにそれぞれindexごと4分割した数を入れる.

train_x['period'] = np.arange(0, len(train_rej_oct)) // (len(train_rej_oct) // 4) 



# 0から3に収める.

train_x['period'] = np.clip(train_x['period'], 0, 3)



# testはとりあえずperiodをすべて4とする.

test['period'] = 4



train_data = lgb.Dataset(train_x, label=train_rej_oct["y"])



param = {'num_leaves': 31, 'objective': 'binary'}

param['verbose'] = -1

param['metric'] = 'auc'



# 時系列考慮した交差検証法を行う.

va_period_list = [1, 2, 3]

for va_period in va_period_list:

    is_tr =train_x['period'] < va_period

    is_va=train_x['period'] == va_period

    tr_x, va_x = train_x[is_tr], train_x[is_va]

    tr_y, va_y = train_y[is_tr], train_y[is_va]

    

    train_data = lgb.Dataset(tr_x, tr_y)

    valid_data = lgb.Dataset(va_x, va_y)

    

    #cvでなく、データセットを指定して直接trainを呼ぶ

    lgb.train(param, train_data, 50, valid_sets=[train_data, valid_data], verbose_eval=True)    



#以上を、train0→train1 train0～1→train2　train0～2→train3　の予測を繰り返す

test_x = test.drop(["index", "duration","day","month"], axis=1)

y_pred = bst.predict(test_x)

sub = pd.read_csv("../input/ds2019uec-task1/sample_submission.csv")

sub["pred"] = y_pred

sub.to_csv("sample_lgbm.csv", index=False)
train_y_pred = bst.predict(train_x)

plt.hist(train_y_pred)
plt.hist(y_pred)
#変数重要度をチェック



lgb.plot_importance(bst, figsize=(12, 6))