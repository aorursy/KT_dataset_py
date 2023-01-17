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
##ライブラリインポート

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression



##データ読み込み

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
##表示設定変更

pd.set_option('display.max_rows',100)

pd.set_option('display.max_columns',100)



train.describe
train
train.corr()
##家の材料及び質と価格　相関係数=0.790982



a = train[["OverallQual"]].values

b = train["SalePrice"].values



# 散布図を描画

plt.scatter(a,b)

plt.xlabel('OverallQual')

plt.ylabel('SalePrice')



# 表示

plt.show()
##築年数と価格 相関係数=0.522897



a = train[["YearBuilt"]].values



# 散布図を描画

plt.scatter(a,b)

plt.xlabel('YearBuilt')

plt.ylabel('SalePrice')



# 表示

plt.show()
#欠損値確認

train.isnull().sum()
#データタイプの確認

train.dtypes
##訓練データ中の欠損値を含む項目の確認

train_nan = train.isnull().sum()

train_nan = train_nan[train_nan > 0]

train_nan
# 文字列をラベル化した数値に変換する為のライブラリをインポート

from sklearn.preprocessing import LabelEncoder



# データタイプがobjectの列の値をラベル化した数値に変換

for i in range(train.shape[1]):

    if train.iloc[:,i].dtypes == object:

        lbl = LabelEncoder()

        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))

        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))

        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

        

train
# Label Encodeされた後の、SaleConditionカラムにセットされている値の一覧

train["SaleCondition"].unique()
##訓練データ中の欠損値を含む項目の確認

train_nan = train.isnull().sum()

train_nan = train_nan[train_nan > 0]

train_nan
##訓練データ中の欠損を中央値で補完する

train = train.fillna(train.median())
##訓練データ中の欠損値を含む項目の確認

train_nan = train.isnull().sum()

train_nan = train_nan[train_nan > 0]

train_nan
## テストデータの欠損も同様にする

test = test.fillna(train.median())
##テストデータについても欠損値を確認

test_nan = test.isnull().sum()

test_nan = test_nan[test_nan > 0]

test_nan
##提出用にIDを保存

train_ID = train['Id']

test_ID = test['Id']



##訓練用にデータをx,yに分割

train_y = train['SalePrice']

train_x = train.drop(['Id','SalePrice'],axis = 1)

test_x = test.drop('Id',axis = 1)



##訓練データとテストデータの結合

Xmat = pd.concat([train_x, test_x])
##トータル面積という新たな変数を追加する

Xmat["TotalSF"] = Xmat["TotalBsmtSF"] + Xmat["1stFlrSF"] + Xmat["2ndFlrSF"]
## その他纏められる変数の追加

## 1部屋あたりの面積

Xmat["FeetPerRoom"] = Xmat["TotalSF"] / Xmat["TotRmsAbvGrd"]



#建築した年とリフォームした年の合計

Xmat['YearBuiltAndRemod'] = Xmat['YearBuilt'] + Xmat['YearRemodAdd']



#バスルームの合計面積

Xmat['Total_Bathrooms'] = (Xmat['FullBath'] + (0.5 * Xmat['HalfBath']) +

                               Xmat['BsmtFullBath'] + (0.5 * Xmat['BsmtHalfBath']))



#縁側の合計面積

Xmat['Total_porch_sf'] = (Xmat['OpenPorchSF'] + Xmat['3SsnPorch'] +

                              Xmat['EnclosedPorch'] + Xmat['ScreenPorch'] +

                              Xmat['WoodDeckSF'])
##延べ面積と価格の相関図

tot = Xmat.iloc[:train.shape[0],:]



a = tot[["TotalSF"]].values

a = np.transpose(a) ##転置してる

b = train["SalePrice"].values



# 散布図を描画

plt.scatter(a,b)

plt.xlabel('TotalSF')

plt.ylabel('SalePrice')



# 表示

plt.show()
## 延べ面積と価格の相関係数

np.corrcoef(a, b)
Xmat
sns.distplot(train_y)

plt.show()
train_y.head()
train_y.sort_values(ascending=False).head()
# 対数計算を実施

# 数字のばらつき、偏りを小さくする

train_y = np.log(train_y)



sns.distplot(train_y)

plt.show()
##trainとtestに再分割

train_x = Xmat.iloc[:train.shape[0],:]

test_x = Xmat.iloc[train.shape[0]:,:]



from sklearn.ensemble import RandomForestRegressor as RFR

rf = RFR(n_estimators=100, max_features='auto',random_state=111)

rf.fit(train_x,train_y)

ranking = np.argsort(-rf.feature_importances_)

X_name=train_x.columns



fim1=[]

fim2=[]

for i in range(len(ranking)):

    fim1.append(X_name[ranking[i]])

    fim2.append(rf.feature_importances_[ranking[i]])



#fim1:特徴量の名前 fim2:その特徴量の重要度　大きい順に並んでいる
#ここで表示できる行，列を多くしている(pandasのみ)

pd.set_option('display.max_rows',300)

pd.set_option('display.max_columns',300)

#fim1とfim2を結合，大きな意味はないが表で見ることができる

fim1d=pd.DataFrame(fim1,columns=["Label"])

fim2d=pd.DataFrame(fim2,columns=["Im"])

fim=pd.concat([fim1d,fim2d],axis=1)

fim
# use the top 49 features only (importance>1e-03)

train_x = train_x.iloc[:,ranking[:49]]

test_x = test_x.iloc[:,ranking[:49]]
# ヒートマップにして相関を可視化する。その前準備

# 算出した相関係数を相関が高い順に上位10個のデータを表示

For_corr = train_x

For_corr["SalePrice"] = train_y



corrmat = For_corr.corr()

corrmat
#correlation matrix

sns.set(font_scale=0.8)

f, ax = plt.subplots(figsize=(20, 15))

sns.heatmap(corrmat,  cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 6});
## 相関が大きい変数は片方を除去し、多重共線性を回避する

#train_x = train_x.drop(['SalePrice','GrLivArea','GarageCars','1stFlrSF','YearBuilt','YearRemodAdd'],axis = 1)

#test_x = test_x.drop(['GrLivArea','GarageCars','1stFlrSF','YearBuilt','YearRemodAdd'],axis = 1)



train_x = train_x.drop(['SalePrice','YearBuilt'],axis = 1)

test_x = test_x.drop(['YearBuilt'],axis = 1)
train_x
test_x
# z-scoreにて標準化

# (値 - 平均) / 標準偏差

train_x = (train_x - train_x.mean()) / train_x.std()

test_x = (test_x - test_x.mean()) / test_x.std()
# 外れ値の除去にあたり、大きく家の売値に影響している上位2項目に関してグラフ化する。

# まず総面積と売値



a = train_x[["TotalSF"]].values

b = train_y



# 散布図を描画

plt.scatter(a,b)



# 表示

plt.show()
# 次に家のクオリティと売値



a = train_x[["OverallQual"]].values



# 散布図を描画

plt.scatter(a,b)



# 表示

plt.show()
# ちなみに3番目の家の築年数+モデル年数と売値

# 外れ値と認識し辛いため3位以降は外れ値無視



a = train_x[["YearBuiltAndRemod"]].values



# 散布図を描画

plt.scatter(a,b)



# 表示

plt.show()
# 外れ値を削除

Xmat = train_x

Xmat['SalePrice'] = train_y



# TotalSFのグラフより

Xmat = Xmat.drop(index = Xmat[(Xmat['TotalSF'] > 5) & (Xmat['SalePrice'] < 12.5)].index)
#　確認

a = Xmat[["TotalSF"]].values

b = Xmat["SalePrice"].values



# 散布図を描画

plt.scatter(a,b)



# 表示

plt.show()
#　確認

a = Xmat[["OverallQual"]].values

b = Xmat["SalePrice"].values



# 散布図を描画

plt.scatter(a,b)



# 表示

plt.show()
# recover

train_y = Xmat['SalePrice']

train_x = Xmat.drop(['SalePrice'], axis=1)
# アルゴリズムに線形回帰(Linear Regression)を採用

slr = LinearRegression()



# fit関数で学習開始

slr.fit(train_x,train_y)



# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力

# 偏回帰係数はscikit-learnのcoefで取得

print('傾き：{0}'.format(slr.coef_[0]))



# y切片(直線とy軸との交点)を出力

# 余談：x切片もあり、それは直線とx軸との交点を指す

print('y切片: {0}'.format(slr.intercept_))
##　モデルの評価(平均2乗誤差の算出)

from sklearn.metrics import mean_squared_error



y_train_pred = np.exp(slr.predict(train_x))

mean_squared_error(np.exp(train_y), y_train_pred)
## モデルの評価(決定係数Rの算出)

from sklearn.metrics import r2_score



r2_score(np.exp(train_y), y_train_pred)
## 訓練データの予測値

y_train_pred
# 学習済みのモデルから予測した結果をセット

# logで小さくなった尺度をexpで戻す

y_test_pred = np.exp(slr.predict(test_x))
# 学習済みのモデルから予測した結果を出力

y_test_pred
# submission

submission = pd.DataFrame({

    "Id": test_ID,

    "SalePrice": y_test_pred

})

submission.to_csv('submission.csv', index=False)