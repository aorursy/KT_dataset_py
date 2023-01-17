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
# data wrangling

import numpy as np

import pandas as pd

import pandas_profiling as pdp

from collections import Counter



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.core.display import display



# modeling

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate



# evaluation

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')
display(train.describe(include='all'))
train.head(50)
train_x = train.drop(['Survived'],axis=1)

train_y = train['Survived']

test_x = test.copy()
#train.head()

#train_x.head()

#train_y.head()

#train_x.head()
from sklearn.preprocessing import LabelEncoder



# 変数PassengerIdを除外する

train_x = train_x.drop(['PassengerId'], axis=1)

test_x = test_x.drop(['PassengerId'], axis=1)



# 変数Name, Ticket, Cabinを除外する

train_x = train_x.drop(['Ticket', 'Cabin'], axis=1)

test_x = test_x.drop(['Ticket', 'Cabin'], axis=1)



# それぞれのカテゴリ変数にlabel encodingを適用する

for c in ['Sex', 'Embarked']:

    # 学習データに基づいてどう変換するかを定める

    le = LabelEncoder()

    le.fit(train_x[c].fillna('NA'))



    # 学習データ、テストデータを変換する

    train_x[c] = le.transform(train_x[c].fillna('NA'))

    test_x[c] = le.transform(test_x[c].fillna('NA'))

#train_x.head(50)
train1_x = train_x.copy()

train1_x['Child'] = train_x['Name'].str.contains('Master')*1

#print(train_x.head(20))

test1_x = test_x.copy()

test1_x['Child'] = test1_x['Name'].str.contains('Master')*1

#print(test_x.head(20))



train0_x = train_x.drop(['Name'], axis=1)

train1_x = train1_x.drop(['Name'], axis=1)

test0_x = test_x.drop(['Name'], axis=1)

test1_x = test1_x.drop(['Name'], axis=1)



#print(train0_x.head())

#print(train1_x.head())

#print(test0_x.head())

#print(test1_x.head())

# -----------------------------------

# モデル作成

# -----------------------------------

from xgboost import XGBClassifier



# モデルの作成および学習データを与えての学習

model = XGBClassifier(n_estimators=20, random_state=71)

model.fit(train0_x, train_y)



# テストデータの予測値を確率で出力する

pred0 = model.predict_proba(test0_x)[:, 1]



# テストデータの予測値を二値に変換する

pred_label0 = np.where(pred0 > 0.5, 1, 0)





#child追加モデル

# モデルの作成および学習データを与えての学習

model = XGBClassifier(n_estimators=20, random_state=71)

model.fit(train1_x, train_y)



# テストデータの予測値を確率で出力する

pred1 = model.predict_proba(test1_x)[:, 1]



# テストデータの予測値を二値に変換する

pred_label1 = np.where(pred1 > 0.5, 1, 0)



#print(pred0)

#print(pred1)



# 提出用ファイルの作成

#submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label})

#submission.to_csv('submission_first.csv', index=False)

# スコア：0.7799（本書中の数値と異なる可能性があります）

# -----------------------------------

# バリデーション

# -----------------------------------

from sklearn.metrics import log_loss, accuracy_score

from sklearn.model_selection import KFold



# 各foldのスコアを保存するリスト

scores_accuracy0 = []

scores_logloss0 = []



# クロスバリデーションを行う

# 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す

kf = KFold(n_splits=4, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x):

    # 学習データを学習データとバリデーションデータに分ける

    tr_x, va_x = train0_x.iloc[tr_idx], train0_x.iloc[va_idx]

    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



    # モデルの学習を行う

    model = XGBClassifier(n_estimators=20, random_state=71)

    model.fit(tr_x, tr_y)



    # バリデーションデータの予測値を確率で出力する

    va_pred = model.predict_proba(va_x)[:, 1]



    # バリデーションデータでのスコアを計算する

    logloss = log_loss(va_y, va_pred)

    accuracy = accuracy_score(va_y, va_pred > 0.5)



    # そのfoldのスコアを保存する

    scores_logloss0.append(logloss)

    scores_accuracy0.append(accuracy)



# 各foldのスコアの平均を出力する

logloss0 = np.mean(scores_logloss0)

accuracy0 = np.mean(scores_accuracy0)

print(f'logloss0: {logloss0:.4f}, accuracy0: {accuracy0:.4f}')

# logloss: 0.4270, accuracy: 0.8148（本書中の数値と異なる可能性があります）
#child追加モデル



# -----------------------------------

# バリデーション

# -----------------------------------

from sklearn.metrics import log_loss, accuracy_score

from sklearn.model_selection import KFold



# 各foldのスコアを保存するリスト

scores_accuracy1 = []

scores_logloss1 = []



# クロスバリデーションを行う

# 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す

kf = KFold(n_splits=4, shuffle=True, random_state=71)

for tr_idx, va_idx in kf.split(train_x):

    # 学習データを学習データとバリデーションデータに分ける

    tr_x, va_x = train1_x.iloc[tr_idx], train1_x.iloc[va_idx]

    tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



    # モデルの学習を行う

    model = XGBClassifier(n_estimators=20, random_state=71)

    model.fit(tr_x, tr_y)



    # バリデーションデータの予測値を確率で出力する

    va_pred = model.predict_proba(va_x)[:, 1]



    # バリデーションデータでのスコアを計算する

    logloss = log_loss(va_y, va_pred)

    accuracy = accuracy_score(va_y, va_pred > 0.5)



    # そのfoldのスコアを保存する

    scores_logloss1.append(logloss)

    scores_accuracy1.append(accuracy)



# 各foldのスコアの平均を出力する

logloss1 = np.mean(scores_logloss1)

accuracy1 = np.mean(scores_accuracy1)

print(f'logloss1: {logloss1:.4f}, accuracy1: {accuracy1:.4f}')

# logloss: 0.4270, accuracy: 0.8148（本書中の数値と異なる可能性があります）
#print(kf.split(train_x))
print(np.arange(1,10,0.5).tolist())

np.arange(1,10,0.5).tolist()
print([3,5,7])

print([1.0,2.0,4.0])
#mdv = np.arange(1,10,0.5).tolist()

#mdv
# -----------------------------------

# モデルチューニング

# -----------------------------------

import itertools



mdv = np.arange(1,10).tolist()

mcwv = np.arange(1,10,0.2).tolist()



# チューニング候補とするパラメータを準備する

param_space = {

    'max_depth': mdv,

    'min_child_weight': mcwv

}



# 探索するハイパーパラメータの組み合わせ

param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])



# 各パラメータの組み合わせ、それに対するスコアを保存するリスト

params0 = []

scores0 = []



# 各パラメータの組み合わせごとに、クロスバリデーションで評価を行う

for max_depth, min_child_weight in param_combinations:



    score_folds = []

    # クロスバリデーションを行う

    # 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す

    kf = KFold(n_splits=4, shuffle=True, random_state=123456)

    for tr_idx, va_idx in kf.split(train0_x):

        # 学習データを学習データとバリデーションデータに分ける

        tr_x, va_x = train0_x.iloc[tr_idx], train0_x.iloc[va_idx]

        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



        # モデルの学習を行う

        model = XGBClassifier(n_estimators=20, random_state=71,

                              max_depth=max_depth, min_child_weight=min_child_weight)

        model.fit(tr_x, tr_y)



        # バリデーションデータでのスコアを計算し、保存する

        va_pred = model.predict_proba(va_x)[:, 1]

        logloss = log_loss(va_y, va_pred)

        score_folds.append(logloss)



    # 各foldのスコアを平均する

    score_mean = np.mean(score_folds)



    # パラメータの組み合わせ、それに対するスコアを保存する

    params0.append((max_depth, min_child_weight))

    scores0.append(score_mean)



# 最もスコアが良いものをベストなパラメータとする

best_idx0 = np.argsort(scores0)[0]

best_param0 = params0[best_idx0]

print(f'max_depth0: {best_param0[0]}, min_child_weight0: {best_param0[1]}')

# max_depth=7, min_child_weight=2.0のスコアが最もよかった
#child追加モデル



# -----------------------------------

# モデルチューニング

# -----------------------------------

import itertools



mdv = np.arange(1,10).tolist()

mcwv = np.arange(1,10,0.2).tolist()



# チューニング候補とするパラメータを準備する

param_space = {

    'max_depth': mdv,

    'min_child_weight': mcwv

}



# 探索するハイパーパラメータの組み合わせ

param_combinations = itertools.product(param_space['max_depth'], param_space['min_child_weight'])



# 各パラメータの組み合わせ、それに対するスコアを保存するリスト

params1 = []

scores1 = []



# 各パラメータの組み合わせごとに、クロスバリデーションで評価を行う

for max_depth, min_child_weight in param_combinations:



    score_folds = []

    # クロスバリデーションを行う

    # 学習データを4つに分割し、うち1つをバリデーションデータとすることを、バリデーションデータを変えて繰り返す

    kf = KFold(n_splits=4, shuffle=True, random_state=123456)

    for tr_idx, va_idx in kf.split(train1_x):

        # 学習データを学習データとバリデーションデータに分ける

        tr_x, va_x = train1_x.iloc[tr_idx], train1_x.iloc[va_idx]

        tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]



        # モデルの学習を行う

        model = XGBClassifier(n_estimators=20, random_state=71,

                              max_depth=max_depth, min_child_weight=min_child_weight)

        model.fit(tr_x, tr_y)



        # バリデーションデータでのスコアを計算し、保存する

        va_pred = model.predict_proba(va_x)[:, 1]

        logloss = log_loss(va_y, va_pred)

        score_folds.append(logloss)



    # 各foldのスコアを平均する

    score_mean = np.mean(score_folds)



    # パラメータの組み合わせ、それに対するスコアを保存する

    params1.append((max_depth, min_child_weight))

    scores1.append(score_mean)



# 最もスコアが良いものをベストなパラメータとする

best_idx1 = np.argsort(scores1)[0]

best_param1 = params1[best_idx1]

print(f'max_depth1: {best_param1[0]}, min_child_weight1: {best_param1[1]}')

# max_depth=7, min_child_weight=2.0のスコアが最もよかった
# -----------------------------------

# チューニングした値で実行

# -----------------------------------

from xgboost import XGBClassifier



# モデルの作成および学習データを与えての学習

model = XGBClassifier(n_estimators=20, random_state=71,max_depth=best_param0[0],min_child_weight=best_param0[1])

model.fit(train0_x, train_y)



# テストデータの予測値を確率で出力する

pred = model.predict_proba(test0_x)[:, 1]



# テストデータの予測値を二値に変換する

pred_label10 = np.where(pred > 0.5, 1, 0)



print(pred_label10)



# 提出用ファイルの作成

#submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label10})

#submission.to_csv('submission_first.csv', index=False)

# スコア：0.7799（本書中の数値と異なる可能性があります）
#child追加モデル



# -----------------------------------

# チューニングした値で実行

# -----------------------------------

from xgboost import XGBClassifier



# モデルの作成および学習データを与えての学習

model = XGBClassifier(n_estimators=20, random_state=71,max_depth=best_param1[0],min_child_weight=best_param1[1])

model.fit(train1_x, train_y)



# テストデータの予測値を確率で出力する

pred = model.predict_proba(test1_x)[:, 1]



# テストデータの予測値を二値に変換する

pred_label11 = np.where(pred > 0.5, 1, 0)



print(pred_label11)



# 提出用ファイルの作成

submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': pred_label11})

submission.to_csv('submission_first.csv', index=False)

# スコア：0.7799（本書中の数値と異なる可能性があります）
#print(param_combinations)
# -----------------------------------

# ロジスティック回帰用の特徴量の作成

# -----------------------------------

from sklearn.preprocessing import OneHotEncoder



# 元データをコピーする

train_x2 = train.drop(['Survived'], axis=1)

test_x2 = test.copy()



# 変数PassengerIdを除外する

train_x2 = train_x2.drop(['PassengerId'], axis=1)

test_x2 = test_x2.drop(['PassengerId'], axis=1)



# 変数Name, Ticket, Cabinを除外する

train_x2 = train_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)

test_x2 = test_x2.drop(['Name', 'Ticket', 'Cabin'], axis=1)



# one-hot encodingを行う

cat_cols = ['Sex', 'Embarked', 'Pclass']

ohe = OneHotEncoder(categories='auto', sparse=False)

ohe.fit(train_x2[cat_cols].fillna('NA'))



# one-hot encodingのダミー変数の列名を作成する

ohe_columns = []

for i, c in enumerate(cat_cols):

    ohe_columns += [f'{c}_{v}' for v in ohe.categories_[i]]



# one-hot encodingによる変換を行う

ohe_train_x2 = pd.DataFrame(ohe.transform(train_x2[cat_cols].fillna('NA')), columns=ohe_columns)

ohe_test_x2 = pd.DataFrame(ohe.transform(test_x2[cat_cols].fillna('NA')), columns=ohe_columns)



# one-hot encoding済みの変数を除外する

train_x2 = train_x2.drop(cat_cols, axis=1)

test_x2 = test_x2.drop(cat_cols, axis=1)



# one-hot encodingで変換された変数を結合する

train_x2 = pd.concat([train_x2, ohe_train_x2], axis=1)

test_x2 = pd.concat([test_x2, ohe_test_x2], axis=1)



# 数値変数の欠損値を学習データの平均で埋める

num_cols = ['Age', 'SibSp', 'Parch', 'Fare']

for col in num_cols:

    train_x2[col].fillna(train_x2[col].mean(), inplace=True)

    test_x2[col].fillna(train_x2[col].mean(), inplace=True)



# 変数Fareを対数変換する

train_x2['Fare'] = np.log1p(train_x2['Fare'])

test_x2['Fare'] = np.log1p(test_x2['Fare'])
# -----------------------------------

# アンサンブル

# -----------------------------------

from sklearn.linear_model import LogisticRegression



# xgboostモデル

model_xgb = XGBClassifier(n_estimators=20, random_state=71)

model_xgb.fit(train_x, train_y)

pred_xgb = model_xgb.predict_proba(test_x)[:, 1]



# ロジスティック回帰モデル

# xgboostモデルとは異なる特徴量を入れる必要があるので、別途train_x2, test_x2を作成した

model_lr = LogisticRegression(solver='lbfgs', max_iter=300)

model_lr.fit(train_x2, train_y)

pred_lr = model_lr.predict_proba(test_x2)[:, 1]



# 予測値の加重平均をとる

pred = pred_xgb * 0.8 + pred_lr * 0.2

pred_label = np.where(pred > 0.5, 1, 0)