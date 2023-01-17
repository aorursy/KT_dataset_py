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
# zipファイルの解凍（カレントディレクトリに出力される）

import zipfile



with zipfile.ZipFile("/kaggle/input/restaurant-revenue-prediction/test.csv.zip") as zf:

    zf.extractall()

with zipfile.ZipFile("/kaggle/input/restaurant-revenue-prediction/train.csv.zip") as zf:

    zf.extractall()

     
# ファイルをデータフレームに読み込み

df_train = pd.read_csv("train.csv")

df_test = pd.read_csv("test.csv")

df_submission = pd.read_csv("/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv")
df_test
# データの欠損値の確認



# df_train.isna().sum()

# df_test.isna().sum()



# →　学習データ・テストデータ共に欠損値無し
# データ型の確認



# pd.DataFrame(df_train).dtypes
# 全ての項目の相関係数

corrmat = df_train.corr()

# corrmat
# 学習データの目的変数を先に外しておく



y_train = df_train["revenue"]

del df_train["revenue"]
# 学習データ・テストデータ一緒に前処理してしまうために結合



df_all = pd.concat([df_train, df_test], axis=0)   # axis=0 : 縦方向に結合
# OpenDateを分解



# 文字列をtimestamp型に変換

df_all["Open Date"] = pd.to_datetime(df_all["Open Date"])



df_all["Year"] = df_all["Open Date"].dt.year

df_all["Month"] = df_all["Open Date"].dt.month

df_all["Day"] = df_all["Open Date"].dt.day
# カテゴリデータを数値に変換

# カテゴリデータ：City, City Group, Type

from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df_all["City"] = le.fit_transform(df_all["City"])

df_all["City Group"] = le.fit_transform(df_all["City Group"])

df_all["Type"] = le.fit_transform(df_all["Type"])

# df_all
# データの前処理が終わったので再度学習データ・テストデータに分割



df_train_fin = df_all.iloc[:df_train.shape[0]]   # 上からdf_trainにある行だけを抽出

df_test_fin = df_all.iloc[df_train.shape[0]:]   # 下からdf_testにある行だけを抽出
from sklearn.ensemble import RandomForestRegressor



# 説明変数の設定 　IDとOpenDate以外すべて使う

out_columns = ["Id", "Open Date"]

columns = []



for i in df_train_fin.columns:

    if i not in out_columns:

        columns.append(i)



x_train = df_train_fin[columns]





# 学習

rfr = RandomForestRegressor(

    n_estimators=200, 

    max_depth=5, 

    max_features=0.5, 

    random_state=449,

    n_jobs=-1

)

rfr.fit(x_train, y_train)





# モデルの精度

rfr.score(x_train, y_train)
pred = rfr.predict(df_test_fin[columns])
df_submission
df_submission['Prediction'] = pred

df_submission.to_csv('/kaggle/working/RandamForest_submission01.csv', index=False)
ll