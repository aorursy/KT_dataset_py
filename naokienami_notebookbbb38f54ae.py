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
import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline

import seaborn as sns



from sklearn.metrics import roc_auc_score, mean_squared_error, mean_squared_log_error, log_loss



from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm

from sklearn import preprocessing as pp

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, PowerTransformer, quantile_transform



import lightgbm as lgb

from lightgbm import LGBMClassifier, LGBMRegressor



import eli5

from eli5.sklearn import PermutationImportance



from sklearn.svm import SVC

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor



import itertools

import math

import re
# データロード

df_train_read = pd.read_csv('../input/exam-for-students20200923/train.csv', index_col=0) 

df_test_read = pd.read_csv('../input/exam-for-students20200923/test.csv',index_col=0)

df_country_read = pd.read_csv('../input/exam-for-students20200923/country_info.csv')
df_train_read.shape
df_train_read.head(5)
df_test_read.shape
df_test_read.head(5)
df_country_read.shape
df_country_read.head(5)
# トレーニングデータの分布確認

plt.hist(df_train_read["ConvertedSalary"])
# 評価指標がRMSLEのため対数変換

df_train_read["ConvertedSalary"] = np.log1p(df_train_read["ConvertedSalary"])
# トレーニングデータの分布再確認

plt.hist(df_train_read["ConvertedSalary"])
#後で分割するけど、前処理用に結合

df_train_read["partition"] = "train"

df_test_read["partition"] = "test"

df_train_test= pd.concat([df_train_read, df_test_read])
df_train_test.shape
# print(df_country_read['Service']).str.replace(',', '.')
# Excelで確認したところ、country_info.csvの数値データがカンマになっているため前処理

# object型を抜き取る

li = []

for i in df_country_read.columns:

    print(i,":",df_country_read[i].dtype)

    if df_country_read[i].dtype == "object":

        li.append(i)

        

# Country, Regionは対象外

print("-------------------------------------------")

print(li)

li.remove('Country')

li.remove('Region')



# 確認

print("-------------------------------------------")

print(li)



# カンマをドットに変換

for i in li:

    df_country_read[i] = df_country_read[i].str.replace(",",".")

    df_country_read[i] = df_country_read[i].astype("float64")



# 確認

print("-------------------------------------------")

for i in df_country_read.columns:

    print(i,":",df_country_read[i].dtype)
# country_infoの結合

df_train_test_country = df_train_test.merge(df_country_read, on='Country', how='left').set_index(df_train_test.index)
# 確認

df_train_test_country.head(5)
pd.set_option("display.max_rows",1000)

pd.set_option("display.max_columns",1000)
# 各列のユニーク数を見る

df_train_test_country.nunique()
# 各列のデータタイプを見る

for i in df_train_test_country.columns:

    print(i,":",df_train_test_country[i].dtype)
# Excelより、テキスト→順序込みの数値で表現すべきと思った列の変換

# Age

print(df_train_test_country["Age"].value_counts())

dic = {"Under 18 years old":1,"18 - 24 years old":2,"25 - 34 years old":3,"35 - 44 years old":4,"45 - 54 years old":5,"55 - 64 years old":6,"65 years or older":7}

df_train_test_country["Age"].replace(dic, inplace=True)

print(df_train_test_country["Age"].value_counts())
# CompanySize

print(df_train_test_country["CompanySize"].value_counts())

dic = {"Fewer than 10 employees":1,"10 to 19 employees":2,"20 to 99 employees":3,"100 to 499 employees":4,"500 to 999 employees":5,"1,000 to 4,999 employees":6,"5,000 to 9,999 employees":7,"10,000 or more employees":8}

df_train_test_country["CompanySize"].replace(dic, inplace=True)

print(df_train_test_country["CompanySize"].value_counts())
# YearsCoding

print(df_train_test_country["YearsCoding"].value_counts())

dic = {"0-2 years":1,"3-5 years":2,"6-8 years":3,"9-11 years":4,"12-14 years":5,"15-17 years":6,"18-20 years":7,"21-23 years":8,"24-26 years":9,"27-29 years":10,"30 or more years":11}

df_train_test_country["YearsCoding"].replace(dic, inplace=True)

print(df_train_test_country["YearsCoding"].value_counts())
# YearsCodingProf

print(df_train_test_country["YearsCodingProf"].value_counts())

dic = {"0-2 years":1,"3-5 years":2,"6-8 years":3,"9-11 years":4,"12-14 years":5,"15-17 years":6,"18-20 years":7,"21-23 years":8,"24-26 years":9,"27-29 years":10,"30 or more years":11}

df_train_test_country["YearsCodingProf"].replace(dic, inplace=True)

print(df_train_test_country["YearsCodingProf"].value_counts())
# ordinalエンコーディングの準備

li = df_train_test_country.columns

ordinal = []

for i in li:

    print(i,":",df_train_test_country[i].dtype)

    if df_train_test_country[i].dtype == "object":

        ordinal.append(i)



# partition列は除外

ordinal.remove("partition")
# object型は一括でordinalエンコーディング

oe = OrdinalEncoder(cols=ordinal, return_df=False)

df_train_test_country[ordinal] = oe.fit_transform(df_train_test_country[ordinal])
#欠損値補完

df_train_test_country.fillna(-1, inplace=True)
# トレーニングデータとテストデータの分割

X_train_org = df_train_test_country[df_train_test_country['partition'] == "train"]

X_test_org = df_train_test_country[df_train_test_country['partition'] == "test"]



print(X_train_org.shape)

print(X_test_org.shape)



# partition列は不要

X_train_org.drop(['partition'], axis=1, inplace=True)

X_test_org.drop(['partition'], axis=1, inplace=True)



print(X_train_org.shape)

print(X_test_org.shape)
# データの分割

y_train = X_train_org["ConvertedSalary"]

X_train = X_train_org.drop(["ConvertedSalary"], axis=1)

X_test = X_test_org.drop(["ConvertedSalary"], axis=1)

X_train.sort_index(axis=1, ascending=False)

X_test.sort_index(axis=1, ascending=False)
# 特徴量の相関を見てみる（train）

corr = X_train.corr()

cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

corr.style.background_gradient(cmap='coolwarm')
# 良くならなかった

# 相関の高い特徴量(目安0.9)を削除

# X_train.drop(["Agriculture","GDP ($ per capita)","Industry"], axis=1, inplace=True)

# X_test.drop(["Agriculture","GDP ($ per capita)","Industry"], axis=1, inplace=True)
# ベースラインとして、LGBM単体で実装時の名残

# X_train1,X_test1,y_train1,y_test1 = train_test_split(X_train,y_train)



# # lightGBMで進める

# model = lgb.LGBMRegressor()

# model.fit(X_train1, y_train1)



# # 予測実行

# pred = model.predict(X_test1)

# # 確認

# pred[1:5]



# # RMSLEを戻す

# pred1 = np.expm1(pred)

# # 確認

# pred1[1:5]



# pred_test = model.predict(X_test)

# pred_test1 = np.expm1(pred_test)
# KFoldを試す

y_pred_test_li = []



# シード値を何通りか試してみる

for seed in [1,2,3,4,5]:

# for seed in [1,2,3,4,6]:

# for seed in [10,20,30,40,50]:

# for seed in [100,200,300,400,500]:

    kf = KFold(n_splits=5, random_state=seed, shuffle=True)

    for i, (train_ix, test_ix) in enumerate(kf.split(X_train, y_train)):

        # トレーニングデータと検証データに分割

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]



        # lightGBMで回してみる

        clf = LGBMRegressor(

#             learning_rate = 0.05,

            learning_rate = 0.1,

#             num_leaves=64,

            num_leaves=31,

            colsample_bytree=0.9,

            subsample=0.9,

            n_estimators=9999,

            random_state=seed)

        clf.fit(X_train_, y_train_, early_stopping_rounds=50, eval_metric='rmse', eval_set=[(X_val, y_val)], verbose=100)



        # 検証データに対して予測(RMSLEを戻す)

        y_pred = np.expm1(clf.predict(X_val))

        # 負の値はあり得ないので0に

        y_pred[y_pred < 0] = 0



        # テストデータに対して予測(RMSLEを戻す)

        y_pred_test = np.expm1(clf.predict(X_test))

        # 負の値はあり得ないので0に

        y_pred_test[y_pred_test < 0] = 0

        

        # 終わったら代入

        y_pred_test_li.append(y_pred_test)



# 平均でアンサンブル

y_pred = np.mean(y_pred_test_li, axis=0)
# 提出

submission = pd.read_csv('../input/exam-for-students20200923/sample_submission.csv', index_col=0)



# 整数型に変換してから代入

submission['ConvertedSalary'] = y_pred.astype("int64")

submission.to_csv('submission.csv')