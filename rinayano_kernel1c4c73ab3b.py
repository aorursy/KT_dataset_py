# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import pymc3 as pm

from sklearn.preprocessing import PowerTransformer

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

from sklearn.model_selection import GridSearchCV

import lightgbm as lgb

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#ファイルをインプット

##以下、ロジスティック回帰分析

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

train_label = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

test_label = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

#データの読み込み

#学習データ

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

#テストデータ

test_x = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



#########################################################################################

'''参考にしたもの

https://medium-s.jp/kaggle-house-prices/

https://qiita.com/Yuta33553297/items/3a2a5134d99b8154e988

https://www.kaggle.com/c/zillow-prize-1

'''

#変数処理



#物件の広さを合計した変数を作成

train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test_x["TotalSF"] = test_x["1stFlrSF"] + test_x["2ndFlrSF"] + test_x["TotalBsmtSF"]



#家の材質・完成度：OverallQual

#外れ値を除外する

train = train.drop(train[(train['OverallQual']<5) & (train['SalePrice']>200000)].index)

train = train.drop(train[(train['OverallQual']<10) & (train['SalePrice']>500000)].index)

train = train.drop(train[(train['OverallQual']==10) & (train['SalePrice']<200000)].index)

train = train.drop(train[(train['OverallQual']==10) & (train['SalePrice']>700000)].index)



#築年数:YearRemodAddとYearBuiltを統合

#改装した場合、部屋などは新築と同様にきれいになる

#改築している場合、統合

#2011年のデータなので、2011から引く



train['YearBuilt'][train['YearBuilt']!=train['YearRemodAdd']]=train['YearRemodAdd']

test_x['YearBuilt'][test_x['YearBuilt']!=test_x['YearRemodAdd']]=test_x['YearRemodAdd']

train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>410000)].index)

train = train.drop(train[(train['YearBuilt']<1970) & (train['SalePrice']>300000)].index)

train['YearBuilt'] = 2011 - train['YearBuilt']

test_x['YearBuilt'] = 2011 - test_x['YearBuilt']





#物件の広さを合計した変数を作成

#1階の広さ+2階の広さ+地下室の廣さ

train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]

test_x["TotalSF"] = test_x["1stFlrSF"] + test_x["2ndFlrSF"] + test_x["TotalBsmtSF"]



########################################################################################

#データ統合



#学習データを目的変数とそれ以外に分ける

train_x = train.drop("SalePrice",axis=1)

train_y = train["SalePrice"]



#学習データとテストデータを統合

all_data = pd.concat([train_x,test_x],axis=0,sort=True)



#IDのカラムは不必要なので別の変数に格納

train_ID = train['Id']

test_ID = test_x['Id']



all_data.drop("Id", axis = 1, inplace = True)



########################################################################################

#それぞれのデータのサイズ

#train_x: (1446, 81),train_y: (1446,),test_x: (1459, 81),all_data: (2905, 80),train_ID:(1446,),test_ID:(1459,)



########################################################################################

#欠損値処理

#欠損値の数

all_data_na = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

# 欠損値があるカラムをリスト化

na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist()

#隣接した道路の長さ（LotFrontage）の欠損値の補完

#同じ地区の場合に存在するほかの物件と同じとする

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#欠損値が存在するかつfloat型のリストを作成

float_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "float64"].index.tolist()

#欠損値が存在するかつobject型のリストを作成

obj_list = all_data[na_col_list].dtypes[all_data[na_col_list].dtypes == "object"].index.tolist()

#float型の場合は欠損値を0で置換

all_data[float_list] = all_data[float_list].fillna(0)

#object型の場合は欠損値を"None"で置換

all_data[obj_list] = all_data[obj_list].fillna("None")



#######################################################################################

#カテゴリ変数化

#数値的意味をもたないデータをカテゴリ変数に変換する

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)



#######################################################################################

#目的変数を正規分布へ近づける

#目的変数の対数log(x+1)をとる

train_y = np.log1p(train_y)

#######################################################################################

#説明変数も対数変換

#歪度>0.5は対数変換

#数値の説明変数のリストを作成

num_feats = all_data.dtypes[all_data.dtypes != "object" ].index

#各説明変数の歪度を計算

skewed_feats = all_data[num_feats].apply(lambda x: x.skew()).sort_values(ascending = False)

#歪度の絶対値が0.5より大きい変数だけに絞る

skewed_feats_over = skewed_feats[abs(skewed_feats) > 0.5].index

#Yeo-Johnson変換

pt = PowerTransformer()

pt.fit(all_data[skewed_feats_over])

#変換後のデータで各列を置換

all_data[skewed_feats_over] = pt.transform(all_data[skewed_feats_over])

#各説明変数の歪度を計算

skewed_feats_fixed = all_data[skewed_feats_over].apply(lambda x: x.skew()).sort_values(ascending = False)

#######################################################################################

#変数の追加

#特徴量に1部屋あたりの面積を追加

all_data["FeetPerRoom"] =  all_data["TotalSF"]/all_data["TotRmsAbvGrd"]

#建築した年とリフォームした年の合計

all_data['YearBuiltAndRemod']=all_data['YearBuilt']+all_data['YearRemodAdd']

#バスルームの合計面積

all_data['Total_Bathrooms'] = (all_data['FullBath'] + (0.5 * all_data['HalfBath']) + all_data['BsmtFullBath'] + (0.5 * all_data['BsmtHalfBath']))

#縁側の合計面積

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] + all_data['3SsnPorch'] + all_data['EnclosedPorch'] + all_data['ScreenPorch'] + all_data['WoodDeckSF'])

#プールの有無

all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

#2階の有無

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

#ガレージの有無

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

#地下室の有無

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

#暖炉の有無

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

#########################################################################################

#カテゴリ変数を処理

#各カラムのデータ型を確認

#カテゴリ変数(object):46

#カテゴリ変数となっているカラムを取り出す

cal_list = all_data.dtypes[all_data.dtypes=="object"].index.tolist()

#カテゴリ変数をget_dummiesによるone-hot-encodingを行う

all_data = pd.get_dummies(all_data,columns=cal_list)

#########################################################################################

#統合したデータの分割

train_x = all_data.iloc[:train_x.shape[0],:].reset_index(drop=True)

test_x = all_data.iloc[train_x.shape[0]:,:].reset_index(drop=True)

#########################################################################################

#データサイズ

#train_x: (1446, 349) test_x: (1459, 349)

#########################################################################################

# データの分割

#学習データを学習用データと評価用データに分割

train_x, valid_x, train_y, valid_y = train_test_split(train_x,train_y,test_size=0.3,random_state=0)

#特徴量と目的変数をxgboostのデータ構造に変換する

dtrain = xgb.DMatrix(train_x, label=train_y)

dvalid = xgb.DMatrix(valid_x,label=valid_y)

#パラメータを指定してGBDT

num_round = 5000

evallist = [(dvalid, 'eval'), (dtrain, 'train')]

evals_result = {}

#パラメータ

param = {'max_depth': 3,'eta': 0.01,'objective': 'reg:squarederror',}

#学習の実行

bst = xgb.train(param, dtrain,num_round,evallist,evals_result=evals_result,early_stopping_rounds=1000)

# 一定ラウンド回しても改善が見込めない場合は学習を打ち切る

#########################################################################################

#提出データの作成

dtest = xgb.DMatrix(test_x)

my_submission = pd.DataFrame()

my_submission["Id"] = test_ID

my_submission["SalePrice"] = np.exp(bst.predict(dtest))

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)