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
import numpy as np

import pandas as pd

import category_encoders as ce

import re

from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import mean_squared_log_error

from sklearn.feature_extraction.text import CountVectorizer



import lightgbm as lgb

from lightgbm import LGBMClassifier

from lightgbm import LGBMRegressor



from scipy import sparse, hstack



# datetime

import datetime



import numpy as np

import matplotlib.pyplot as plt



# 以下はこのノートブックのみ



# Stop Words: NLTKから英語のstop wordsを読み込み

from nltk.corpus import stopwords



# 可視化のためのセットです。

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline



# CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
# 読み込み

df_train = pd.read_csv('/kaggle/input/train.csv', index_col = 0)

df_test = pd.read_csv('/kaggle/input/test.csv', index_col = 0)



# trainデータをX, yへ分離

df_X_train = df_train.drop('ConvertedSalary', axis=1)

df_y_train = df_train[['ConvertedSalary']]



# testデータは元々Xのみだが、名前だけ変えておきます。

df_X_test = df_test
# X_trainを確認(始めの5行)

df_X_train.head()
# y_trainを確認(始めの5行)

df_y_train.head()
# X_testを確認(始めの5行)

df_X_test.head()
# CurrencyはCurrencySymbolとかぶるので削除

df_X_train.drop(['Currency'], axis=1, inplace=True)

df_X_test.drop(['Currency'], axis=1, inplace=True)
# とりあえず、全てのオブジェクト属性についてOrdinalEncoderを使用する。



# カテゴリ変数として扱うカラムのカラム名をリスト化

# オブジェクトカラムを抽出 (object)

list_cols_cat = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'object':

        list_cols_cat.append(i)



list_cols_cat.remove('DevType')

list_cols_cat.remove('CommunicationTools')

print(list_cols_cat)
# エンコーダを作成

ce_oe = ce.OrdinalEncoder(cols=list_cols_cat, handle_unknown='impute')



# カテゴリ変数をOrdinal Encodingし、新たなデータフレームを作成

df_X_train_prep = ce_oe.fit_transform(df_X_train) # df_X_trainをエンコード

df_X_test_prep = ce_oe.fit_transform(df_X_test) # df_X_testをエンコード
# Oridinal Encoding後のdf_X_train_prepを確認(始めの5行)

df_X_train_prep.head()
# Oridinal Encoding後のdf_X_test_prepを確認(始めの5行)

df_X_test_prep.head()
# 念のため、NaNが含まれるカラムを確認します。



# NaNが含まれる行数を列ごとにカウント

ser_X_train_isnan = df_X_train_prep.isnull().sum()

ser_y_train_isnan = df_y_train.isnull().sum()

ser_X_test_isnan = df_X_test_prep.isnull().sum()



# NaNが含まれるカラム名を抽出しリスト化

list_X_train_colname_isnan = ser_X_train_isnan.index[ser_X_train_isnan > 0].tolist()

list_y_train_colname_isnan = ser_y_train_isnan.index[ser_y_train_isnan > 0].tolist()

list_X_test_colname_isnan = ser_X_test_isnan.index[ser_X_test_isnan > 0].tolist()



# 確認

print('X_trainで欠損があるカラム', list_X_train_colname_isnan)

print('y_trainで欠損があるカラム', list_y_train_colname_isnan)

print('X_testで欠損があるカラム', list_X_test_colname_isnan)
# カラムにおける欠損値を、それぞれの中央値で穴埋めする

df_X_train_prep.fillna(df_X_train_prep.median(), inplace=True)

df_y_train.fillna(df_y_train.median(), inplace=True)

df_X_test_prep.fillna(df_X_test_prep.median(), inplace=True)
# NaNがなくなったことを確認します。



# NaNが含まれる行数を列ごとにカウント

ser_X_train_isnan = df_X_train_prep.isnull().sum()

ser_y_train_isnan = df_y_train.isnull().sum()

ser_X_test_isnan = df_X_test_prep.isnull().sum()



# NaNが含まれるカラム名を抽出しリスト化

list_X_train_colname_isnan = ser_X_train_isnan.index[ser_X_train_isnan > 0].tolist()

list_y_train_colname_isnan = ser_y_train_isnan.index[ser_y_train_isnan > 0].tolist()

list_X_test_colname_isnan = ser_X_test_isnan.index[ser_X_test_isnan > 0].tolist()



# 確認

print('X_trainで欠損があるカラム', list_X_train_colname_isnan)

print('y_trainで欠損があるカラム', list_y_train_colname_isnan)

print('X_testで欠損があるカラム', list_X_test_colname_isnan)
# 数値のカラムを指定

# 数値カラムを抽出 (float or int)

list_cols_num = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'float64' or df_X_train[i].dtype == 'int64':

        list_cols_num.append(i)

        

print(list_cols_num)
# 学習データに基づいて複数カラムの標準化を定義

scaler = StandardScaler()

scaler.fit(df_X_train_prep[list_cols_num])



# 変換後のデータで各列を置換

df_X_train_prep[list_cols_num] = scaler.transform(df_X_train_prep[list_cols_num])

df_X_test_prep[list_cols_num] = scaler.transform(df_X_test_prep[list_cols_num])
# 標準化後のdf_X_train_prepを確認(始めの5行)

df_X_train_prep.head()
# 標準化後のdf_X_test_prepを確認(始めの5行)

df_X_test_prep.head()
# テキストカラムをリストで抜き出し

X_train_text_DevType = df_X_train_prep.DevType

X_test_text_DevType = df_X_test_prep.DevType



# 欠損値をNaNという言葉で埋める

X_train_text_DevType.fillna('NaN', inplace=True)

X_test_text_DevType.fillna('NaN', inplace=True)
# 置換

X_train_text_DevType = X_train_text_DevType.str.replace(';', ' ')

X_train_text_DevType = X_train_text_DevType.str.replace('(', ' ')

X_train_text_DevType = X_train_text_DevType.str.replace(')', ' ')



X_test_text_DevType = X_test_text_DevType.str.replace(';', ' ')

X_test_text_DevType = X_test_text_DevType.str.replace('(', ' ')

X_test_text_DevType = X_test_text_DevType.str.replace(')', ' ')
# 全てのDevTypeをTfIdfでベクトル化

vec_all = CountVectorizer(max_features=100000)



# DevTypeは全て使う

DevType_all = pd.concat([X_train_text_DevType, X_test_text_DevType])



# 全てのDevTypeをTfIdfでベクトル化

vec_all.fit_transform(DevType_all)
# X_train_text用ベクタライザーの指定

# 辞書はvec_allで抽出したものを使う。

vec_train = CountVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)



# X_train_textをベクトル化

X_train_text_DevType_tfidf = vec_train.fit_transform(X_train_text_DevType)
X_train_text_DevType_tfidf
# X_test_text用ベクタライザーの指定

# 辞書は辞書はvec_allで抽出したものを使う。

vec_test = CountVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)



# X_test_textをベクトル化

X_test_text_DevType_tfidf = vec_test.fit_transform(X_test_text_DevType)
X_test_text_DevType_tfidf
# DevTypeをデータフレームからドロップ

df_X_train_prep.drop(['DevType'], axis=1, inplace=True)

df_X_test_prep.drop(['DevType'], axis=1, inplace=True)
# テキストカラムをリストで抜き出し

X_train_text_CommunicationTools = df_X_train_prep.CommunicationTools

X_test_text_CommunicationTools = df_X_test_prep.CommunicationTools



# 欠損値をNaNという言葉で埋める

X_train_text_CommunicationTools.fillna('NaN', inplace=True)

X_test_text_CommunicationTools.fillna('NaN', inplace=True)



# 置換

X_train_text_CommunicationTools = X_train_text_CommunicationTools.str.replace(';', ' ')

X_train_text_CommunicationTools = X_train_text_CommunicationTools.str.replace('(', ' ')

X_train_text_CommunicationTools = X_train_text_CommunicationTools.str.replace(')', ' ')



X_test_text_CommunicationTools = X_test_text_CommunicationTools.str.replace(';', ' ')

X_test_text_CommunicationTools = X_test_text_CommunicationTools.str.replace('(', ' ')

X_test_text_CommunicationTools = X_test_text_CommunicationTools.str.replace(')', ' ')



# 全てのCommunicationToolsをTfIdfでベクトル化

vec_all = CountVectorizer(max_features=100000)



# CommunicationToolsは全て使う

CommunicationTools_all = pd.concat([X_train_text_CommunicationTools, X_test_text_CommunicationTools])



# 全てのCommunicationToolsをTfIdfでベクトル化

vec_all.fit_transform(CommunicationTools_all)



# X_train_text用ベクタライザーの指定

# 辞書はvec_allで抽出したものを使う。

vec_train = CountVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)



# X_train_textをベクトル化

X_train_text_CommunicationTools_tfidf = vec_train.fit_transform(X_train_text_CommunicationTools)



# X_test_text用ベクタライザーの指定

# 辞書は辞書はvec_allで抽出したものを使う。

vec_test = CountVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)



# X_test_textをベクトル化

X_test_text_CommunicationTools_tfidf = vec_test.fit_transform(X_test_text_CommunicationTools)



# DevTypeをデータフレームからドロップ

df_X_train_prep.drop(['CommunicationTools'], axis=1, inplace=True)

df_X_test_prep.drop(['CommunicationTools'], axis=1, inplace=True)
# 全てのテキストをstack

X_train_text_tfidf = sparse.hstack([X_train_text_DevType_tfidf, X_train_text_DevType_tfidf])

X_test_text_tfidf = sparse.hstack([X_test_text_DevType_tfidf, X_test_text_DevType_tfidf])
# Xを指定

# スパース行列を作成

X_train_prep_sparse = sparse.csr_matrix(df_X_train_prep.values) # TfIdf以外のdenseをsparseに

X = sparse.hstack([X_train_prep_sparse, X_train_text_tfidf])

# 行方向に圧縮

X = X.tocsr()





# X = df_X_train_prep.values



# yを指定

y = df_y_train.ConvertedSalary.values



# アルゴリズムを指定

# LightGBM のハイパーパラメータ

regressor = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                           importance_type='split', learning_rate=0.05, max_depth=-1,

                           min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                           n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                           random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                           subsample=0.9, subsample_for_bin=200000, subsample_freq=0)



# 層化抽出法における分割数を指定(5分割)

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
# 5回のValidation Scoreを格納するリストを作成

list_cv_rmsle_score = []

# best iteration回数を格納するリストを作成

list_num_best_iteration = []



d = datetime.datetime.today()

print('start:', d)



t = 0



# 5分割層化抽出によるクロスバリデーションを実行

# 各回のスコアを記録

for train_ix, valid_ix in skf.split(X, y):

    X_train, y_train = X[train_ix], y[train_ix]

    X_valid, y_valid = X[valid_ix], y[valid_ix]



    # フィッティング

    # fit_train = classifier.fit(X_train, y_train)

    # LGBM

    fit_train = regressor.fit(X_train, y_train,

                              early_stopping_rounds=200,

                              eval_metric='rmse',

                              eval_set=[(X_valid, y_valid)],

                              verbose=100)

    # 予測を実施

    pred_valid = fit_train.predict(X_valid)

    # 予測が負になったら0とする

    pred_valid = np.where(pred_valid>0, pred_valid, 0)

    # RMSLEを計算

    v_rmsle_score = np.sqrt(mean_squared_log_error(y_valid, pred_valid))

    # スコアを追加

    list_cv_rmsle_score.append(v_rmsle_score)

    

    # best_iteraion回数を記録

    num_best_iteration = fit_train.best_iteration_

    list_num_best_iteration.append(num_best_iteration)

    

    # タイムスタンプをprint

    t = t + 1

    d = datetime.datetime.today()

    print(t, '_finished:', d)

    

    # スコア表示

    print('RMSLEは', v_rmsle_score)

    print('Best Iteration回数は', num_best_iteration)
# Cross Validationスコアを算出

cv_rmsle_score = np.mean(list_cv_rmsle_score)

print('RSMLEは', cv_rmsle_score)



# best iteration回数の平均を算出

av_num_best_iteration = np.mean(list_num_best_iteration).round()

av_num_best_iteration = np.int(av_num_best_iteration)

print('best iteration回数の平均は', av_num_best_iteration)
# タイムスタンプをprint

d = datetime.datetime.today()

print('start:', d)



# 全てのデータで学習

regressor_all = LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

                               importance_type='split', learning_rate=0.05, max_depth=-1,

                               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                               n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

                               random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

                               subsample=0.9, subsample_for_bin=200000, subsample_freq=0,

                               num_iteration = av_num_best_iteration)



fit_all = regressor_all.fit(X, y)



# testに対して予測を実行

# スパース行列を作成

X_test_prep_sparse = sparse.csr_matrix(df_X_test_prep.values) # TfIdf以外のdenseをsparseに

X_test_prep = sparse.hstack([X_test_prep_sparse, X_test_text_tfidf])

# 行方向に圧縮

X_test_prep = X_test_prep.tocsr()



# X_test_prep = df_X_test_prep.values









pred_test = fit_all.predict(X_test_prep)

# 予測が負になったら0とする

pred_test = np.where(pred_test>0, pred_test, 0)





# タイムスタンプをprint

d = datetime.datetime.today()

print('finished:', d)
# sample_submission.csvを読み込み

df_submission = pd.read_csv('/kaggle/input/sample_submission.csv', index_col=0)



# 読み込んだDataFrameに、クラス1に分類される確率の予測結果を代入

df_submission.ConvertedSalary = pred_test



# csvファイルに出力

df_submission.to_csv('submission.csv')
plt.hist(pred_test, bins=100)