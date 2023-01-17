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
#必要なライブラリをインポート

import numpy as np

import pandas as pd

import category_encoders as ce



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfVectorizer



from scipy import sparse, hstack

import datetime
# 読み込み

df_train = pd.read_csv('../input/train.csv', index_col=0)

df_test = pd.read_csv('../input/test.csv', index_col=0)
# 欠損値の多いを抽出

list_cols_null = []

for i in df_train.columns:

    if df_train[i].isnull().sum() > 10000:

        list_cols_null.append(i)

        

print(list_cols_null)
df_train = df_train.drop(list_cols_null, axis=1)
df_test = df_test.drop(list_cols_null, axis=1)
# trainデータをX, yへ分離

df_X_train = df_train.drop('ConvertedSalary', axis=1)

df_y_train = df_train[['ConvertedSalary']]

# testデータは元々Xのみだが、名前だけ変えておきます。

df_X_test = df_test
# 数値カラムを抽出（float or int）

list_cols_num = []

for i in df_X_train.columns:

    if df_X_train[i].dtypes == 'float64' or df_X_train[i].dtypes == 'int':

        list_cols_num.append(i)

    

print(list_cols_num)
# 複数カラムの標準化を定義

#scaler = StandardScaler()

#scaler.fit(df_X_train[list_cols_num])
# 変換後のデータで各列で置換

#df_X_train[list_cols_num] = scaler.transform(df_X_train[list_cols_num])

#df_X_test[list_cols_num] = scaler.transform(df_X_test[list_cols_num])
# カラムの欠損値を、それぞれの中央値で穴埋め

df_X_train.fillna(df_X_train.median(), inplace=True)

df_y_train.fillna(df_y_train.median(), inplace=True)

df_X_test.fillna(df_X_test.median(), inplace=True)
# オブジェクトカラムを抽出

list_cols_cat = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'object':

        list_cols_cat.append(i)

        

print(list_cols_cat)
# DevTypeはテキスト処理で使うので除外

list_cols_cat.remove('DevType')
# エンコーダを作成

oe = ce.OrdinalEncoder(cols=list_cols_cat, handle_unknown='impute')



# カテゴリ変数をOrdinal Encodingし、新たなデータフレームを作成

df_X_train = oe.fit_transform(df_X_train)

df_X_test = oe.fit_transform(df_X_test)
# テキストカラムを抜き出す

X_train_text = df_X_train.DevType

X_test_text = df_X_test.DevType



# 欠損値にNaNで埋める

X_train_text.fillna('NaN', inplace=True)

X_test_text.fillna('NaN', inplace=True)
# 全てのDevTypeをTfIdfでベクトル化

vec_all = TfidfVectorizer(max_features=100000)
# DevTypeはすべて使う

emp_title_all = pd.concat([X_train_text, X_test_text])
# 全てのDevTypeをTfIdfでベクトル化

vec_all.fit_transform(emp_title_all)
# X_train_text用ベクタライザーの指定

# 辞書はvec_allで抽出したものを使う

vec_train = TfidfVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)
# X_train_textをベクトル化

X_train_text_tfidf = vec_train.fit_transform(X_train_text)
# X_test_text用ベクタライザーの指定

# 辞書はvec_allで抽出したものを使う

vec_test = TfidfVectorizer(max_features=100000, vocabulary=vec_all.vocabulary_)
# X_test_textをベクトル化

X_test_text_tfidf = vec_test.fit_transform(X_test_text)
# DevTypeをデータフレームからドロップ

df_X_train.drop(['DevType'], axis=1, inplace=True)

df_X_test.drop(['DevType'], axis=1, inplace=True)
# スパース行列を指定

# Tfidf以外のdenseをsparseに

X_train_sparse = sparse.csc_matrix(df_X_train.values)

X = sparse.hstack([X_train_sparse, X_train_text_tfidf])

#行方向に圧縮

X = X.tocsr()
# yを指定

y = df_y_train.ConvertedSalary.values
#from lightgbm import LGBMClassifier

#clf = LGBMClassifier(n_estimators=9999, random_state=71)
from sklearn.dummy import DummyClassifier

clf = DummyClassifier()

fit_train = clf.fit(X, y)
#classifier = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.71,

 #                          importance_type='split', learning_rate=0.05, max_depth=-1,

 #                          min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

 #                          n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

 #                          random_state=71, reg_alpha=1.0, reg_lambda=1.0, silent=True,

 #                          subsample=0.9, subsample_for_bin=200000, subsample_freq=0)
# 層化抽出法による分割数を指定（5分割）

skf = StratifiedKFold(n_splits=5, random_state=71, shuffle=True)
# 5分割層化抽出によるクロスバリデーションを実行

# 各回のスコアを記録

for train_ix, valid_ix in skf.split(X, y):

    X_train, y_train = X[train_ix], y[train_ix]

    X_valid, y_valid = X[valid_ix], y[valid_ix]

    

    # フィッティング

    fit_train = clf.fit(X_train, y_train)
# スパース行列を指定

# Tfidf以外のdenseをsparseに

X_test_sparse = sparse.csc_matrix(df_X_test.values)

X_test = sparse.hstack([X_test_sparse, X_test_text_tfidf])

#行方向に圧縮

X_test = X_test.tocsr()
pred = fit_train.predict(X_test)

len(pred)
submission = pd.read_csv('../input/sample_submission.csv', index_col=0)
submission.ConvertedSalary = pred

submission
# 予測結果を出力

submission.to_csv('./submission.csv')