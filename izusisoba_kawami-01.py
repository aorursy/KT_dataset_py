##　memo

#

# やりたいこと

# 0 （大前提）基本的な前処理でモデル作る

# 1 評価指標に対してターゲットをあわせる

# 2 行方向のチューニング　学習データに併せる形で不要な行（精度を悪くするような行？）を除外

# 3 列方向のチューニング　学習データに併せる形で不要な列（精度を悪くするような行？）を除外

# 4 交差検定

# 5 その他　極端な値に補正をかけたい

#

# やったこと

# 1のみ、

#　交差検定なしで、100％データで過学習がこわい。時間があったらこの辺を調整する

#

# やりたかったこと→時間が、、pythonをまた学んできます。

# 駅や町のの場所を位置情報から説明変数→最寄り駅からの距離とか

# 説明変数の選定

# とりあえず事前に作っといたパイプラインのパッケージをインストール　1　



import pandas as pd

import numpy as np

from pathlib import Path

import os

import re

from collections import defaultdict

import math

import glob



from scipy import stats



import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns
# とりあえず事前に作っといたパイプラインのパッケージをインストール 2



from sklearn.metrics import roc_auc_score

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

import time

from tqdm import tqdm_notebook as tqdm

import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



from keras.layers import Input, Dense, Dropout, BatchNormalization

from keras.optimizers import Adam

from keras.models import Model

from keras.callbacks import EarlyStopping

from sklearn.linear_model import LogisticRegression



from sklearn.dummy import DummyClassifier



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor
#とりあえずseedを5個準備

seeds = [6, 66, 666, 6666, 66666]
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#表示枠を設定 (もとの説明変数数が34なので、とりあえず50×100)

pd.set_option('display.max_columns', 50)

pd.set_option('display.max_rows', 100)
#目的変数を設定

col_target = 'TradePrice'
path ='/kaggle/input/exam-for-students20200527/'



df_train = pd.read_csv(path + '/train.csv')

df_test = pd.read_csv(path + '/test.csv')



df_city_inf =pd.read_csv(path + '/city_info.csv')

df_station_info =pd.read_csv(path + '/station_info.csv')



df_train['is_train'] = True

df_test['is_train'] = False

df_test[col_target] = -0.5

df_test = df_test[df_train.columns]



df_all = pd.concat([df_train, df_test], axis=0)
#トレーニングデータ

df_all[df_all['is_train']].describe()
#ターゲットわかりにくかったので、

print(np.min(df_all[df_all['is_train']][col_target]))

print(np.mean(df_all[df_all['is_train']][col_target]))

print(np.max(df_all[df_all['is_train']][col_target]))
#テストデータ

df_all[~ (df_all['is_train'])].describe()
# 評価指標がRMSLEなのでターゲットをログ変換してからモデリングするように変換しとく

df_all[col_target] = np.log1p(df_all[col_target])
def count_missing(df_all):

    # 欠損値の数を返す

    df_all['missing_count'] = df_all.isnull().sum(axis=1)

    return df_all
def encode_missing_pattern(df_all):

    # 欠損値の存在する列のフラグを並べ一つの2進数としてエンコード (１つだけなら1,2列あれば11=>3,3列あれば111＝＞7)

    m = df_all.isnull().sum()                  #列の欠損数を取得し

    cols_with_missing = list(m[m != 0].index) #欠損の存在する列リストを作成



    df_all['missing_pattern'] = 0   

    for col in cols_with_missing:

        df_all['missing_pattern'] *= 2

        df_all.loc[df_all[col].isnull(), 'missing_pattern'] += 1 #行指定：指定した列が欠損、列指定：missing_pattern

    

    # ケタが大きくなりすぎるので小さくする

    df_all['missing_pattern'] *= 1e-16

    return df_all
#数値列に対する欠損埋め

def missing_value_impute_numbers(df_all):

    # 数値は-1で埋める

    numeric_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['int64', 'float64']:

            numeric_cols.append(col)

    numeric_cols.remove(col_target)



    for col, v in df_all[numeric_cols].isnull().sum().iteritems():

        if v == 0:

            continue

        df_all.loc[df_all[col].isnull(), col] = -1



    return df_all
#カテゴリ列に対する欠損埋め

def missing_value_impute_categories(df_all):

    # カテゴリはマーカーで埋める

    missing_marker = '__MISSING_VALUE__'

    categorical_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['object']:

            categorical_cols.append(col)



    for col, v in df_all[categorical_cols].isnull().sum().iteritems():

        if v == 0:

            continue

        df_all.loc[df_all[col].isnull(), col] = missing_marker



    return df_all
# カテゴリ型を数値化

def encode_categorical_features(df_all):

    categorical_cols = []

    for col in df_all.columns:

        if df_all[col].dtype in ['object']:

            categorical_cols.append(col) #object型の一覧を取得

        

    for col in categorical_cols:

        df_all[col] = df_all[col].map(df_all[col].value_counts()) #出現数で数値化

        

    return df_all
# 軽く見ておく 1  

df_all.head(20)
# 軽く見ておく 2

#df_all.tail(20)
#行単位の単純欠損数を新規説明変数に

df_all = count_missing(df_all)
#欠損のパターンを新規の説明変数に

df_all = encode_missing_pattern(df_all)
#数値の欠損埋め

df_all = missing_value_impute_numbers(df_all)
#カテゴリの欠損埋め

df_all = missing_value_impute_categories(df_all)
#カテゴリのままだとダメなロジックも多いので 

df_all = encode_categorical_features(df_all)
#前処理結果を確認

df_all[df_all['is_train']].describe()
df_all[~ (df_all['is_train'])].describe()
#IDは消す(バックアップとってから)

df_all_log = df_all.copy()

df_all = df_all.drop(['id'],axis=1)
X_train = df_all[df_all['is_train']].drop(columns=[col_target, 'is_train',])

y_train = df_all[df_all['is_train']][col_target]

X_test = df_all[~ (df_all['is_train'])].drop(columns=[col_target, 'is_train'])
rounds=50

n_splits=2 #時間がかかりすぎる模様。。交差検定を行う場合も2が限界かな？。時間があまったら。。

seed = seeds[3]
#rfr1 = RandomForestRegressor(n_estimators=rounds,random_state=seed,bootstrap=True, criterion='mse', max_depth=None,max_features='auto', max_leaf_nodes=None,min_impurity_split=1e-07, min_samples_leaf=1,min_samples_split=2, min_weight_fraction_leaf=0.0,n_jobs=1, oob_score=True,verbose=0, warm_start=False)
#交差検定でやってみようかと思ったけどめっちゃ時間かかったので、スルー

#scores = []



#kf = KFold(n_splits=n_splits,shuffle=True,random_state=seed)

#for tr_idx,va_idx,in kf.split(X_train):

#    tr_x,va_x = X_train.iloc[tr_idx],X_train.iloc[va_idx]

#    tr_y,va_y = y_train.iloc[tr_idx],y_train.iloc[va_idx]



#    rfr1.fit(tr_x, tr_y)

#    pred  = rfr1.predict(X_test)

#    scores.append(pred)



#pred = sum(scores) / n_splits
#こっからはシード変化の影響だけを50：50で

#データ分割はとりあえず50:50

seed = seeds[0]

print(seed)



X_t = X_train.sample(round(len(X_train)*0.5),random_state=seed)

X_v = X_train.drop(X_t.index).copy()

y_t = y_train.iloc[X_t.index].copy()

y_v = y_train.iloc[X_v.index].copy()



rfr1 = RandomForestRegressor(n_estimators=rounds,random_state=seed)

rfr1.fit(X_t, y_t)



#精度確認

y_pred  = rfr1.predict(X_t)

score = math.sqrt(sum((y_t - y_pred)**2) / (len(y_pred)))

print('学習')

print(score)



y_pred  = rfr1.predict(X_v)

score = math.sqrt(sum((y_v - y_pred)**2) / (len(y_pred)))

print('検証')

print(score)
#こっからはシード変化の影響だけを50：50で

#データ分割はとりあえず50:50

seed = seeds[1]

print(seed)



X_t = X_train.sample(round(len(X_train)*0.5),random_state=seed)

X_v = X_train.drop(X_t.index).copy()

y_t = y_train.iloc[X_t.index].copy()

y_v = y_train.iloc[X_v.index].copy()



rfr2 = RandomForestRegressor(n_estimators=rounds,random_state=seed)

rfr2.fit(X_t, y_t)



print(seed)

#精度確認

y_pred  = rfr2.predict(X_t)

score = math.sqrt(sum((y_t - y_pred)**2) / (len(y_pred)))

print('学習')

print(score)



y_pred  = rfr2.predict(X_v)

score = math.sqrt(sum((y_v - y_pred)**2) / (len(y_pred)))

print('検証')

print(score)
#こっからはシード変化の影響だけを50：50で

#データ分割はとりあえず50:50

seed = seeds[2]

print(seed)



X_t = X_train.sample(round(len(X_train)*0.5),random_state=seed)

X_v = X_train.drop(X_t.index).copy()

y_t = y_train.iloc[X_t.index].copy()

y_v = y_train.iloc[X_v.index].copy()



rfr3 = RandomForestRegressor(n_estimators=rounds,random_state=seed)

rfr3.fit(X_t, y_t)



#精度確認

y_pred  = rfr3.predict(X_t)

score = math.sqrt(sum((y_t - y_pred)**2) / (len(y_pred)))

print('学習')

print(score)



y_pred  = rfr3.predict(X_v)

score = math.sqrt(sum((y_v - y_pred)**2) / (len(y_pred)))

print('検証')

print(score)
#テストデータを予測

pred1  = rfr1.predict(X_test)

pred2  = rfr2.predict(X_test)

pred3  = rfr3.predict(X_test)



pred  = (pred1+pred2+pred3)/3
submission =pd.read_csv(path + '/sample_submission.csv')

submission = submission[['id']]

submission[col_target] = pred
# logの逆変換

submission[col_target] = np.exp(submission[col_target]) - 1
# 丸め(繰り上げ)

submission[col_target] = submission[col_target].apply(math.ceil)
#提出用ファイル作成

submission.to_csv('submission.csv', index=False)