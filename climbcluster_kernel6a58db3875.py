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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# for visualization

import matplotlib.pyplot as plt

import seaborn as sns



# import data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



print(train.dtypes)
# No.1

# ライブラリのインポート

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression



# Jupyter Notebookの中でインライン表示する場合の設定（これが無いと別ウィンドウでグラフが開く）

%matplotlib inline 
# No.2

# データの読み込み

df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
# No.3

# 読み込んだデータの確認

df
# No.4

# 先頭5行の確認

df.head()
# No.5

# XにOverallQual、yにSalePriceをセット

X = df[["OverallQual"]].values

y = df["SalePrice"].values



# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression()



# fit関数でモデル作成

slr.fit(X,y)



# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力

# 偏回帰係数はscikit-learnのcoefで取得

print('傾き：{0}'.format(slr.coef_[0]))



# y切片(直線とy軸との交点)を出力

# 余談：x切片もあり、それは直線とx軸との交点を指す

print('y切片: {0}'.format(slr.intercept_))
# No.6

# 散布図を描画

plt.scatter(X,y)



# 折れ線グラフを描画

plt.plot(X,slr.predict(X),color='red')



# 表示

plt.show()
# No.7

# テストデータの読込

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# No.8

# テストデータの内容確認(評価用のデータなので、SalePriceはない)

df_test.head()
# No.9

# テストデータの OverallQual の値をセット

X_test = df_test[["OverallQual"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)
# No.10

# 学習済みのモデルから予測した結果を出力

y_test_pred
# No.11

# df_testに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

df_test["SalePrice"] = y_test_pred
# No.12

# df_testの先頭5行を確認

df_test.head()
# No.13

# Id, SalePriceの2列だけ表示

df_test[["Id","SalePrice"]].head()
# No.14

# Id, SalePriceの2列だけのファイルに変換

df_test[["Id","SalePrice"]].to_csv("submission.csv",index=False)