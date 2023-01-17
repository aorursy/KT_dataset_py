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
import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PowerTransformer



from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import (

    LinearRegression,

    Ridge,

    Lasso

)

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# 訓練データの読み込み

df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")



# テストデータの読み込み

df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
# 訓練データの先頭5行を確認する

df_train.head()
#カラムの一覧表示

df_train.columns
#基本統計量の表示

df_train.SalePrice.describe()
# SalePriceに対する相関係数を算出

# 算出した相関係数を相関が高い順に上位10個のデータを表示

abs(df_train.corr()['SalePrice']).nlargest(10)
# 算出した相関係数を相関が高い順に上位10個のデータを表示

corr = df_train.corr()

k = 10

cols = corr.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
#物件の広さを表す変数はいくつかあるので一つに統合した新しい変数を導入する

df_train["TotalSF"] = df_train["1stFlrSF"] + df_train["2ndFlrSF"] + df_train["TotalBsmtSF"]

df_test["TotalSF"] = df_test["1stFlrSF"] + df_test["2ndFlrSF"] + df_test["TotalBsmtSF"]



#散布図の作成

plt.figure(figsize=(10, 5))

plt.scatter(df_train["TotalSF"],df_train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
#外れ値を除外する。

df_train = df_train[df_train['TotalSF'] < 7500]

df_train = df_train.drop(df_train[(df_train['TotalSF']>6000) & (df_train['SalePrice']<500000)].index)



#散布図を作成し、外れ値が消去出来たことを確認する。

plt.figure(figsize=(10, 5))

plt.scatter(df_train["TotalSF"],df_train["SalePrice"])

plt.xlabel("TotalSF")

plt.ylabel("SalePrice")
#築年数と物件価格の散布図を作成する。

data = pd.concat([df_train["YearBuilt"],df_train["SalePrice"]],axis=1)

plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#外れ値を除外する。

df_train = df_train.drop(df_train[(df_train['YearBuilt']<2000) & (df_train['SalePrice']>600000)].index)



#外れ値が消去出来ていることを確認する。

data = pd.concat([df_train["YearBuilt"],df_train["SalePrice"]],axis=1)



plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x="YearBuilt",y="SalePrice",data=data)
#OverallQualについても同様に散布図を作成

plt.figure(figsize=(10, 5))

plt.scatter(df_train["OverallQual"],df_train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#外れ値を除外する

df_train = df_train.drop(df_train[(df_train['OverallQual']<5) & (df_train['SalePrice']>200000)].index)

df_train = df_train.drop(df_train[(df_train['OverallQual']<10) & (df_train['SalePrice']>500000)].index)



#グラフを描画する

plt.figure(figsize=(10, 5))

plt.scatter(df_train["OverallQual"],df_train["SalePrice"])

plt.xlabel("OverallQual")

plt.ylabel("SalePrice")
#訓練データには変数としてSalePriceが含まれる。

#まず訓練データをSalePriceとそれ以外の変数に分離する。

train_a = df_train.drop("SalePrice",axis=1)

train_b = df_train["SalePrice"]



#訓練データとテストデータを統合する。

df_all = pd.concat([train_a,df_test],axis=0,sort=True)



#IDの変数は不必要なので分離する。

train_ID = df_train['Id']

test_ID = df_test['Id']



df_all.drop("Id", axis = 1, inplace = True)



#それぞれのデータのサイズを確認

print("train_a: "+str(train_a.shape)) #訓練データ_SalePrice以外

print("train_b: "+str(train_b.shape)) #訓練データ_SalePriceのみ

print("df_test: "+str(df_test.shape)) #テストデータ_SalePrice以外

print("df_all: "+str(df_all.shape))   #訓練データ＋テストデータ_SalePrice以外
#各変数における欠損値の数を確認する。

#欠損値が多い順に表示する。

df_all.isnull().sum()[df_all.isnull().sum()>0].sort_values(ascending=False)
#各変数のデータの形式を確認する

#float64は数値型、objectはカテゴリカル変数である

na_col_list = df_all.isnull().sum()[df_all.isnull().sum()>0].index.tolist()

df_all[na_col_list].dtypes.sort_values() 
#欠損値はプールがないことを示す→欠損値はnoneで置換

df_all["PoolQC"] = df_all["PoolQC"].fillna("None")
#欠損値はその他の機能がないことを示す→欠損値はnoneで置換

df_all["MiscFeature"] = df_all["MiscFeature"].fillna("None")
#同様に欠損値はnoneで置換

cols = ['Alley', "Fence", "FireplaceQu", 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

df_all[cols] = df_all[cols].fillna('None')
#隣接した道路の長さ（LotFrontage）の欠損値の補完

df_all['LotFrontage'] = df_all.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
df_all = df_all.drop(['Utilities'], axis=1)
#NAは一般的を示す

df_all["Functional"] = df_all["Functional"].fillna("Typ")
#欠損値は一つだけ→最頻値で置換

df_all['Electrical'] = df_all['Electrical'].fillna(df_all['Electrical'].mode()[0])
#欠損値は一つだけ→最頻値で置換

df_all['KitchenQual'] = df_all['KitchenQual'].fillna(df_all['KitchenQual'].mode()[0])
#欠損値は一つだけ→最頻値で置換

df_all['Exterior1st'] = df_all['Exterior1st'].fillna(df_all['Exterior1st'].mode()[0])

df_all['Exterior2nd'] = df_all['Exterior2nd'].fillna(df_all['Exterior2nd'].mode()[0])
#欠損値は一つだけ→最頻値で置換

df_all['SaleType'] = df_all['SaleType'].fillna(df_all['SaleType'].mode()[0])
#建物クラスなし→Noneで置換

df_all['MSSubClass'] = df_all['MSSubClass'].fillna("None")
na_col_list = df_all.isnull().sum()[df_all.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

df_all[na_col_list].dtypes.sort_values() #データ型



#欠損値が存在するかつfloat型のリストを作成

float_list = df_all[na_col_list].dtypes[df_all[na_col_list].dtypes == "float64"].index.tolist()



#欠損値が存在するかつobject型のリストを作成

obj_list = df_all[na_col_list].dtypes[df_all[na_col_list].dtypes == "object"].index.tolist()



#float型の場合は欠損値を0で置換

df_all[float_list] = df_all[float_list].fillna(0)



#object型の場合は欠損値を"None"で置換

df_all[obj_list] = df_all[obj_list].fillna("None")



#欠損値が全て置換できているか確認

df_all.isnull().sum()[df_all.isnull().sum() > 0]
#1部屋あたりの面積

df_all["FeetPerRoom"] =  df_all["TotalSF"]/df_all["TotRmsAbvGrd"]



#建築した年とリフォームした年の合計

df_all['YearBuiltAndRemod']=df_all['YearBuilt']+df_all['YearRemodAdd']



#バスルームの合計面積

df_all['Total_Bathrooms'] = (df_all['FullBath'] + (0.5 * df_all['HalfBath']) +

                               df_all['BsmtFullBath'] + (0.5 * df_all['BsmtHalfBath']))



#縁側の合計面積

df_all['Total_porch_sf'] = (df_all['OpenPorchSF'] + df_all['3SsnPorch'] +

                              df_all['EnclosedPorch'] + df_all['ScreenPorch'] +

                              df_all['WoodDeckSF'])



#プールの有無

df_all['haspool'] = df_all['PoolArea'].apply(lambda x: 1 if x > 0 else 0)



#2階の有無

df_all['has2ndfloor'] = df_all['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)



#ガレージの有無

df_all['hasgarage'] = df_all['GarageArea'].apply(lambda x: 1 if x > 0 else 0)



#地下室の有無

df_all['hasbsmt'] = df_all['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)



#暖炉の有無

df_all['hasfireplace'] = df_all['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
cols = ["MSSubClass", "YrSold", 'MoSold']

df_all[cols] = df_all[cols].astype(str)

df_all[cols] = df_all[cols].astype(str)
#分布を可視化

plt.figure(figsize=(10, 5))

sns.distplot(train_b)
#目的変数の対数log(x+1)をとる

train_b = np.log1p(train_b)



#分布を可視化

plt.figure(figsize=(10, 5))

sns.distplot(train_b)



#正規分布の導入

from scipy.stats import norm



#ヒストグラムに対数をとったSalePriceと正規分布を表示

sns.distplot(train_b, fit = norm)
#数値の説明変数のリストを作成

num_feats = df_all.dtypes[df_all.dtypes != "object" ].index



#各説明変数の歪度を計算

skewed_feats = df_all[num_feats].apply(lambda x: x.skew()).sort_values(ascending = False)



#グラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats.index, y=skewed_feats)
#歪度の絶対値が0.5より大きい変数だけに絞る

skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index



#各変数の最小値を表示

for i in skewed_feats_over:

    print(min(df_all[i]))
#最小値のリストに0が存在するため、Yeo-Johnson変換を適用する。

#Yeo-Johnson変換

pt = PowerTransformer()

pt.fit(df_all[skewed_feats_over])



#変換後のデータで各列を置換

df_all[skewed_feats_over] = pt.transform(df_all[skewed_feats_over])



#各説明変数の歪度を計算

skewed_feats_fixed = df_all[skewed_feats_over].apply(lambda x: x.skew()).sort_values(ascending = False)



#グラフ化

plt.figure(figsize=(20,10))

plt.xticks(rotation='90')

sns.barplot(x=skewed_feats_fixed.index, y=skewed_feats_fixed)
#各カラムのデータ型を確認

df_all.dtypes.value_counts()
#カテゴリ変数となっているカラムを取り出す

cal_list = df_all.dtypes[df_all.dtypes=="object"].index.tolist()



#学習データにおけるカテゴリ変数のデータ数を確認

train_a[cal_list].info()
#カテゴリ変数をget_dummiesによるone-hot-encodingを行う。

df_all = pd.get_dummies(df_all,columns=cal_list)



#サイズを確認。

df_all.shape
#学習データとテストデータに再分割

train_a = df_all.iloc[:train_a.shape[0],:].reset_index(drop=True)

df_test = df_all.iloc[df_train.shape[0]:,:].reset_index(drop=True)



#サイズを確認

print("train_a: "+str(train_a.shape))

print("df_test: "+str(df_test.shape))
from sklearn.linear_model import ElasticNet



scaler = StandardScaler()  #スケーリング

param_grid = [0.1, 0.01, 0.009, 0.005, 0.001] #パラメータグリッド

cnt = 0

for alpha in param_grid:

    ls = ElasticNet(alpha=alpha) #EsasticNet回帰モデル

    pipeline = make_pipeline(scaler, ls) #パイプライン生成

    X_train, X_test, y_train, y_test = train_test_split(train_a, train_b, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    train_rmse = np.sqrt(mean_squared_error(y_train, pipeline.predict(X_train)))

    test_rmse = np.sqrt(mean_squared_error(y_test, pipeline.predict(X_test)))

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    elif best_score > test_rmse:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    else:

        pass

    cnt = cnt + 1

    

print('alpha : ' + str(best_param))

print('test score is : ' +str(best_score))
plt.subplots_adjust(wspace=0.4)

plt.subplot(121)

plt.scatter(np.exp(y_train),np.exp(best_estimator.predict(X_train)))

plt.subplot(122)

plt.scatter(np.exp(y_test),np.exp(best_estimator.predict(X_test)))
# 提出用データ生成

ls = ElasticNet(alpha = 0.01)

pipeline = make_pipeline(scaler, ls)

pipeline.fit(train_a,train_b)

test_SalePrice = pd.DataFrame(np.exp(pipeline.predict(df_test)),columns=['SalePrice'])

df_test_Id = pd.DataFrame(test_ID,columns=['Id'])

pd.concat([df_test_Id, test_SalePrice],axis=1).to_csv('submission.csv',index=False)