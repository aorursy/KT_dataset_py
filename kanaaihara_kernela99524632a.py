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
#データ加工・処理・分析ライブラリ
import numpy as np
import numpy.random as random
import scipy as sp
import pandas as pd


#可視化ライブラリ
import matplotlib.pyplot as plt
import matplotlib as mp1
import seaborn as sns
%matplotlib inline

#機械学習ライブラリ
import sklearn
import sklearn.preprocessing as pp
import sklearn.model_selection as ms
import sklearn.metrics as metrics
import json

from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
#少数第3位まで表示
%precision 3


#データの読み込み
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #テストデータ
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #訓練データ
alldata = pd.concat([train,test],axis=0).reset_index(drop=True)
print("Train data shape:",train.shape)
print("Test data shape:",test.shape)
train.head()
test.head()
 # !相関係数を算出
corrmat = train.corr()
corrmat
# 算出した相関係数を相関が高い順に上位10個のデータを表示

# ヒートマップに表示させるカラムの数
n = 10

# SalePriceとの相関が大きい上位10個のカラム名を取得
cols = corrmat.nlargest(n, 'SalePrice')['SalePrice'].index

# SalePriceとの相関が大きい上位10個のカラムを対象に相関を算出
# .T(Trancepose[転置行列])を行う理由は、corrcoefで相関を算出する際に、各カラムの値を行毎にまとめなければならない為
cm = np.corrcoef(train[cols].values.T)

# ヒートマップのフォントサイズを指定
sns.set(font_scale=1.25)

# 算出した相関データをヒートマップで表示
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# corrmat.nlargest(n, 'SalePrice') : SalePriceの値が大きい順にkの行数だけ抽出した行と全列の行列データ
# corrmat.nlargest(n, 'SalePrice')['SalePrice'] : SalePriceの値が大きい順にnの行数だけ抽出した抽出した行とSalePrice列だけ抽出したデータ
# corrmat.nlargest(n, 'SalePrice')['SalePrice'].index : 行の項目を抽出。データの中身はIndex(['SalePrice', 'OverallQual', ... , 'YearBuilt'], dtype='object')
# cols.values : カラムの値(SalesPrice, OverallQual, ...)
#家の材質・完成度と物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train["GrLivArea"],train["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
 # トレーニングデータの欠損値数
train_null = train.isnull().sum()
train_null[train_null>0].sort_values(ascending=False)
print('データ型の確認(型変換前)\n{}\n'.format(train.dtypes))
#物件の広さを合計した変数を作成
train["TotalSF"] = train["GrLivArea"] + train["GarageArea"] + train["TotalBsmtSF"] + train["1stFlrSF"] + train["2ndFlrSF"] 
test["TotalSF"] =  test["GrLivArea"] + test["GarageArea"] + test["TotalBsmtSF"]+test["1stFlrSF"] + test["2ndFlrSF"] 

#今回使うSalePriceとTotalSFとOverallQualのデータに置き換える
train_data = train[["SalePrice","GrLivArea","GarageArea","TotalBsmtSF","1stFlrSF","2ndFlrSF"]]
#データ分割（訓練データとテストデータ）のためのインポート
from sklearn.model_selection import train_test_split

#重回帰のモデル構築のためのインポート
from sklearn.linear_model import LinearRegression

#目的変数にSalePriceを指定、説明変数にそれ以外を指定
X = train_data.drop("SalePrice",axis=1)
y = train_data["SalePrice"]

#訓練データと検証データvalificationに分ける
train_X,val_X,train_y,val_y = train_test_split(X,y,test_size=0.5,random_state=0)

train_X.head()
train_y.head()

#重回帰クラスの初期化と学習
model = LinearRegression()
model.fit(train_X,train_y)


#決定係数を表示
print('決定係数(train):{:.3f}'.format(model.score(train_X,train_y)))
print('決定係数(test):{:.3f}'.format(model.score(val_X,val_y)))

#回帰係数と切片を表示
print('\n回帰係数\n{}'.format(pd.Series(model.coef_,index=X.columns)))
print('切片:{:.3f}'.format(model.intercept_))
#予測値の算出
train_y_pred = model.predict(train_X) #トレーニングデータでの予測
val_y_pred = model.predict(val_X) #検証用データでの予測
#グラフを描画する
plt.scatter(train_y, train_y_pred)
plt.scatter(val_y, val_y_pred)
plt.show()
# 元のテストデータ から test_X を生成
test_X = test[["GrLivArea","GarageArea","TotalBsmtSF","1stFlrSF","2ndFlrSF"]]

# test_X に nan があるので処理する
# ここでは仮に 0 に変換しておきます。
test_X = test_X.fillna(0)

# test_y を予測する
test_y = model.predict(test_X)

# 元のテストデータに上記の結果を格納する
test['SalePrice'] = model.predict(test_X)

# 表示
test.head()
test[['Id', 'SalePrice']].to_csv('submission.csv', index=False)

