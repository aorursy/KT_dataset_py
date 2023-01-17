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
#ライブラリの読み込み

import numpy as np

import pandas as pd

#グラフ作成用

from matplotlib import pyplot as plt



import seaborn as sns



from sklearn.preprocessing import PowerTransformer

#プロットを3D表示するためのプログラム

from mpl_toolkits.mplot3d import Axes3D

#線形回帰モデルの使用

from sklearn.linear_model import LinearRegression
#データの読み込み

#練習データ

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

#テストデータ

test_x = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#学習データの変数を確認

train.columns
#変数個数確認

print (len(set(train)))
#目的変数である家の価格の要約統計量を表示する

train["SalePrice"].describe()

#目的変数である家の価格のヒストグラムを表示する

plt.figure(figsize=(20, 10))

sns.distplot(train['SalePrice'])
#IDは不必要なので別の変数に格納

train_ID = train['Id']

test_ID = test_x['Id']



#IDのデータの削除及びTrue代入

train.drop("Id", axis = 1, inplace = True)

test_x.drop("Id", axis = 1, inplace = True)
#学習データとテストデータを統合

all_data = pd.concat([train,test_x],axis=0,sort=True)



#それぞれのデータのサイズの確認

print("train_x: "+str(train_x.shape))

print("test_x: "+str(test_x.shape))

print("all_data: "+str(all_data.shape))
print(train.info())

print(test_x.info())
#データの欠損値を確認する

all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

all_data_na
#欠損値の数をグラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=all_data_na.index, y=all_data_na)
# 欠損値があるカラムをリスト化

na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()



#欠損があるカラムのデータ型を確認

all_data[na_col_list].dtypes.sort_values()
#欠損値が存在するかつfloat型のリストを作成#隣接した道路の長さ（LotFrontage）の欠損値の補完

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#欠損値が存在するかつfloat型のリストを作成

float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()



#欠損値が存在するかつobject型のリストを作成

obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()



#float型の場合は欠損値を0で置換

all_data[float_list] = all_data[float_list].fillna(0)



#object型の場合は欠損値を"None"で置換

all_data[obj_list] = all_data[obj_list].fillna("None")

#欠損値が全て置換できているか確認

all_data.isnull().sum()[all_data.isnull().sum() > 0]
#学習データとテストデータに再分割

train = all_data.iloc[:train.shape[0],:].reset_index(drop=True)

test_x = all_data.iloc[train.shape[0]:,:].reset_index(drop=True)

#サイズを確認

print("train: "+str(train.shape))

print("test_x: "+str(test_x.shape))
#物件の広さを合計した変数を作成。物件の大きさと値段の関係が見れる。

train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test_x["TotalSF"] = test_x["1stFlrSF"] + test_x["2ndFlrSF"] + train["TotalBsmtSF"]



#特徴量に1部屋あたりの面積を追加。

train["FeetPerRoom"] =  train["TotalSF"]/train["TotRmsAbvGrd"]

test_x["FeetPerRoom"] =  test_x["TotalSF"]/test_x["TotRmsAbvGrd"]



#建築した年とリフォームした年の合計

train['YearBuiltAndRemod']=train['YearBuilt']+train['YearRemodAdd']

test_x['YearBuiltAndRemod']=test_x['YearBuilt']+test_x['YearRemodAdd']



#バスルームの合計面積

train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) +

                               train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))

test_x['Total_Bathrooms'] = (test_x['FullBath'] + (0.5 * test_x['HalfBath']) +

                               test_x['BsmtFullBath'] + (0.5 * test_x['BsmtHalfBath']))



#縁側の合計面積

train['Total_porch_sf'] = (train['OpenPorchSF'] + train['3SsnPorch'] +

                              train['EnclosedPorch'] + train['ScreenPorch'] +

                              train['WoodDeckSF'])

test_x['Total_porch_sf'] = (test_x['OpenPorchSF'] + test_x['3SsnPorch'] +

                              test_x['EnclosedPorch'] + test_x['ScreenPorch'] +

                              test_x['WoodDeckSF'])





#プールの有無

train['haspool'] = train['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

test_x['haspool'] = test_x['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



#2階の有無

train['has2ndfloor'] = train['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

test_x['has2ndfloor'] = test_x['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



#ガレージの有無

train['hasgarage'] = train['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

test_x['hasgarage'] = test_x['GarageArea'].apply(lambda x: 1 if x > 0 else 0)



#地下室の有無

train['hasbsmt'] = train['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

test_x['hasbsmt'] = test_x['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



#暖炉の有無

train['hasfireplace'] = train['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

test_x['hasfireplace'] = test_x['Fireplaces'].apply(lambda x: 1 if x  > 0 else 0)
#データ分析

# 相関係数を算出

corrmat = train.corr()

corrmat
# 算出した相関係数を相関が高い順に上位10個のデータを表示



# ヒートマップに表示させるカラムの数

k = 10



# SalesPriceとの相関が大きい上位10個のカラム名を取得

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index



# SalesPriceとの相関が大きい上位10個のカラムを対象に相関を算出

# .T(Trancepose[転置行列])を行う理由は、corrcoefで相関を算出する際に、各カラムの値を行毎にまとめなければならない為

cm = np.corrcoef(train[cols].values.T)



# ヒートマップのフォントサイズを指定

sns.set(font_scale=1.25)



# 算出した相関データをヒートマップで表示

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

# 算出した相関係数を相関が低い順に上位10個のデータを表示



# ヒートマップに表示させるカラムの数

k = 10



# SalesPriceとの相関が低い10個のカラム名を取得

cols = corrmat.nsmallest(k, 'SalePrice')['SalePrice'].index



# SalesPriceとの相関が低い上位10個のカラムを対象に相関を算出

# .T(Trancepose[転置行列])を行う理由は、corrcoefで相関を算出する際に、各カラムの値を行毎にまとめなければならない為

cm = np.corrcoef(train[cols].values.T)



# ヒートマップのフォントサイズを指定

sns.set(font_scale=1.25)



# 算出した相関データをヒートマップで表示

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

# 散布図の表示

sns.set()

cols = ['SalePrice', 'TotalSF', 'OverallQual', 'GrLivArea', 'GarageCars', 'YearBuiltAndRemod']

sns.pairplot(train[cols], size = 2.5)

plt.show()
#外れ値の処理

train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<300000)].index)

train = train.drop(train[(train['YearBuiltAndRemod']>3950) & (train['SalePrice']>600000)].index)

train = train.drop(train[(train['YearBuiltAndRemod']<3900) & (train['SalePrice']>400000)].index)
# 散布図の表示

sns.set()

cols = ['SalePrice', 'TotalSF', 'OverallQual', 'GarageCars', 'YearBuiltAndRemod']

sns.pairplot(train[cols], size = 2.5)

plt.show()
# XにGrLivArea、yにSalePriceをセット

X = train[[ "TotalSF", "OverallQual","GarageCars", "YearBuiltAndRemod"]].values

y = train["SalePrice"].values



# アルゴリズムに線形回帰(Linear Regression)

slr = LinearRegression()



# fit関数で学習開始

slr.fit(X,y)



# 偏回帰係数(回帰分析において得られる回帰方程式の各説明変数の係数)を出力

# 偏回帰係数はscikit-learnのcoefで取得

print('傾き：{0}'.format(slr.coef_))

a1, a2,a3,a4 = slr.coef_



# y切片(直線とy軸との交点)を出力

# 余談：x切片もあり、それは直線とx軸との交点を指す

print('y切片: {0}'.format(slr.intercept_))

b = slr.intercept_
# テストデータのTotalSF,OverallQual,GarageCars,YearBuiltAndRemodの値をセット

X_test = test_x[["TotalSF", "OverallQual", "GarageCars", "YearBuiltAndRemod"]].values



# 学習済みのモデルから予測した結果をセット

y_test_pred = slr.predict(X_test)



# 学習済みのモデルから予測した結果を出力

y_test_pred
# test_xに SalePrice カラムを追加し、学習済みのモデルから予測した結果をセット

test_x["SalePrice"] = y_test_pred



#Idも入れる

test_x["Id"] = test_ID
# Id, SalePriceの2列だけ表示

test_x[["Id","SalePrice"]].head()
# Id, SalePriceの2列だけのファイルに変換

test_x[["Id","SalePrice"]].to_csv("submission.csv",index=False)