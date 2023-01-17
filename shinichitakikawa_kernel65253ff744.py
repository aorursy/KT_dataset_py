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
# category_encodersを追加でインストールする

!pip install category_encoders

!pip install lightgbm
#定番モジュール

import pandas as pd

import numpy as np

import scipy as sp

from pandas import DataFrame, Series #DataFrame, Seriesはこの方法でないと入らない

import os



# 可視化

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

import locale



#エンコーディング

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

from sklearn.model_selection import GroupKFold

import category_encoders as ce

from sklearn.feature_extraction.text import TfidfVectorizer

import math



#モデリング

import lightgbm as lgb

from lightgbm import LGBMRegressor



#表示する最大列数を変更

pd.set_option('display.max_columns', 100) 

 

# 表示する最大行数を変更 

pd.set_option('display.max_rows', 100)
import pandas as pd

country_info = pd.read_csv("../input/exam-for-students20200129/country_info.csv")

sample_submission = pd.read_csv("../input/exam-for-students20200129/sample_submission.csv")

survey_dictionary = pd.read_csv("../input/exam-for-students20200129/survey_dictionary.csv")

df_test = pd.read_csv("../input/exam-for-students20200129/test.csv")

df_train = pd.read_csv("../input/exam-for-students20200129/train.csv")



#df_train #42893 rows × 114 columns

#df_test #4809 rows × 113 columns
#データを統合 ※後で分割するので、学習用とテスト用に分けられるようにフラグを付けておく

df_train['train'] = 1

df_test['train'] = 0
#データ統合準備　特徴量とターゲットを分けておく ターゲットは対数変換する

y_train = np.log(df_train['ConvertedSalary'] + 1)

X_train = df_train.drop(['ConvertedSalary'], axis=1)

X_test = df_test
#データ統合（特徴量のみ、データ加工を同時に行う）

#X_all 47702 rows × 114 columns (42893 + 4809 = 47702)

X_all = pd.concat([X_train, X_test])

X_all
# DataFrameの各列のデータ型を確認

pd.set_option('display.max_rows', 500)

print(X_all.dtypes)
# 各列の欠損値の数を調べる（全カラムが表示される）

X_all.isnull().sum()
# 数値型のカラムすべてを標準化

num_cols = []

for col in X_all.columns:

    if X_all[col].dtype == 'int64' or X_all[col].dtype == 'float64':

        num_cols.append(col)



scaler = StandardScaler()

scaler.fit(X_all[num_cols])
#object型のみの特徴量を抽出してリスト化しておく

a = []

for i in X_all.columns:

    if X_all[i].dtype == 'object':

        a.append(i)

        print(i)
#object型の特徴量に対してラベルエンコーディングを施す

for b in a:

   summary = X_all[b].value_counts()

   X_all[b] = X_all[b].map(summary)
#欠損値の個数を特徴量として追加する　欠損があること自体に傾向があると思われる

X_all['missing_amnt'] = X_all.isnull().sum(axis=1)
# 欠損値は中央値で補完する

X_all.fillna(X_all.median())
# トレーニングデータ・テストデータに分割

X_train = X_all[X_all['train'] == 1]

X_test = X_all[X_all['train'] == 0]



X_train.drop(['train'], axis=1, inplace=True)

X_test.drop(['train'], axis=1, inplace=True)
#データの中身確認 ((42893, 114), (4809, 114))

X_train.shape, X_test.shape
def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))
y_pred_test_buf = []



for seed in [71, 10, 20, 30, 40]:

    kf = KFold(n_splits=5, random_state=seed, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(kf.split(X_train, y_train)):

        # トレーニングデータ・検証データに分割

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        # トレーニングデータからモデルを作成

        clf = LGBMRegressor(

            learning_rate = 0.1,

            num_leaves=31,

            colsample_bytree=0.9,

            subsample=0.9,

            n_estimators=9999,

            random_state=seed)

        clf.fit(X_train_, y_train_, early_stopping_rounds=50, eval_metric='rmse', eval_set=[(X_val, y_val)], verbose=100)



        # 検証データに対して予測

        y_pred = np.exp(clf.predict(X_val)) -1

        y_pred[y_pred < 0] = 0

        score = rmsle(np.exp(y_val) -1, y_pred)

        print('CV Score of Fold_%d is %f' % (i, score))



        # テストデータに対して予測

        y_pred_test = np.exp(clf.predict(X_test)) -1

        y_pred_test[y_pred_test < 0] = 0

        y_pred_test_buf.append(y_pred_test)



y_pred_test_mean = np.mean(y_pred_test_buf, axis=0)
submission = pd.read_csv("../input/exam-for-students20200129/sample_submission.csv")
submission['ConvertedSalary'] = y_pred_test_mean
submission.to_csv('submission.csv',index=False)