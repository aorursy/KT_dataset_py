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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_x = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
train.describe
train.columns
print(train.shape)
print(train.dtypes.value_counts())
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
test_x.isnull().sum()[test_x.isnull().sum()>0].sort_values(ascending=False)
# IDを残す
train_ID = train['Id']
test_ID = test_x['Id']

train = train.drop('Id', axis=1)
test_x = test_x.drop('Id', axis=1)

# 欠損値があるカラムをリスト化
train_na_col_list = train.isnull().sum()[train.isnull().sum()>0].index.tolist()
test_x_na_col_list = test_x.isnull().sum()[test_x.isnull().sum()>0].index.tolist()

#欠損値が存在するかつfloat型のリストを作成
float_list_train = train[train_na_col_list].dtypes[train[train_na_col_list].dtypes == "float64"].index.tolist()
float_list_test_x = test_x[test_x_na_col_list].dtypes[test_x[test_x_na_col_list].dtypes == "float64"].index.tolist()

#欠損値が存在するかつobject型のリストを作成
obj_list_train = train[train_na_col_list].dtypes[train[train_na_col_list].dtypes == "object"].index.tolist()
obj_list_test_x = test_x[test_x_na_col_list].dtypes[test_x[test_x_na_col_list].dtypes == "object"].index.tolist()

#float型の場合は欠損値を0で置換
train[float_list_train] = train[float_list_train].fillna(0)
test_x[float_list_test_x] = test_x[float_list_test_x].fillna(0)

#object型の場合は欠損値を"None"で置換
train[obj_list_train] = train[obj_list_train].fillna("None")
test_x[obj_list_test_x] = test_x[obj_list_test_x].fillna("None")


#欠損値が全て置換できているか確認
train.isnull().sum()[train.isnull().sum() > 0]
test_x.isnull().sum()[test_x.isnull().sum() > 0]
#各カラムのデータ型を確認
train.dtypes.value_counts()
test_x.dtypes.value_counts()
#カテゴリ変数となっているカラムを取り出す
train_cal_list = train.dtypes[train.dtypes=="object"].index.tolist()
test_x_cal_list = test_x.dtypes[test_x.dtypes=="object"].index.tolist()

#学習データとテストデータにおけるカテゴリ変数のデータ数を確認
train[train_cal_list].info()
test_x[test_x_cal_list].info()
#カテゴリ変数をget_dummiesによるone-hot-encodingを行う
train = pd.get_dummies(train,columns=train_cal_list)
test_x = pd.get_dummies(test_x,columns=test_x_cal_list)

#サイズを確認
train.shape
test_x.shape
import matplotlib.pyplot as plt
import seaborn as sns
k = 20
corrmat = train.corr()
cols = corrmat.nlargest(k, "SalePrice")["SalePrice"].index
cm = np.corrcoef(train[cols].values.T)
fig, ax = plt.subplots(figsize=(12, 10))
sns.set(font_scale=1.2)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt=".2f", annot_kws={"size": 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
print(train.shape)
#物件の広さを合計した変数を作成
train["TotalSF"] = train["1stFlrSF"] + train["2ndFlrSF"] + train["TotalBsmtSF"]
test_x["TotalSF"] = test_x["1stFlrSF"] + test_x["2ndFlrSF"] + test_x["TotalBsmtSF"]
##バスルームの合計面積
train['Total_Bathrooms'] = (train['FullBath'] + (0.5 * train['HalfBath']) +
                               train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']))
test_x['Total_Bathrooms'] = (test_x['FullBath'] + (0.5 * test_x['HalfBath']) +
                               test_x['BsmtFullBath'] + (0.5 * test_x['BsmtHalfBath']))
plt.figure(figsize=(20, 10))
sns.distplot(train['SalePrice'])
train['SalePrice'] = np.log1p(train['SalePrice'])

#分布を可視化
plt.figure(figsize=(20, 10))
sns.distplot(train['SalePrice'])
plt.figure(figsize=(20, 10))
plt.scatter(train["TotalSF"],train["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['TotalSF']>7500) & (train['SalePrice']<12.5)].index)

#物件の広さと物件価格の散布図を作成
plt.figure(figsize=(20, 10))
plt.scatter(train["TotalSF"],train["SalePrice"])
plt.xlabel("TotalSF")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["YearBuilt"],train["SalePrice"])
plt.xlabel("YearBuilt")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['YearBuilt']<2000) & (train['SalePrice']>13.5)].index)

#グラフを描画する
plt.figure(figsize=(20, 10))
plt.scatter(train["YearBuilt"],train["SalePrice"])
plt.xlabel("YearBuilt")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["OverallQual"],train["SalePrice"])
plt.xlabel("Overallqual")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["GrLivArea"],train["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['GrLivArea']>1000) & (train['SalePrice']<10.7)].index)

#グラフを描画する
plt.figure(figsize=(20, 10))
plt.scatter(train["GrLivArea"],train["SalePrice"])
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["GarageCars"],train["SalePrice"])
plt.xlabel("GarageCars")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['GarageCars']>0.5) & (train['SalePrice']<10.5)].index)

#グラフを描画する
plt.figure(figsize=(20, 10))
plt.scatter(train["GarageCars"],train["SalePrice"])
plt.xlabel("GarageCars")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["FullBath"],train["SalePrice"])
plt.xlabel("FullBath")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['FullBath']>0.5) & (train['SalePrice']<10.7)].index)

#グラフを描画する
plt.figure(figsize=(20, 10))
plt.scatter(train["GarageCars"],train["SalePrice"])
plt.xlabel("GarageCars")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["BsmtQual_Ex"],train["SalePrice"])
plt.xlabel("BsmtQual_Ex")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['BsmtQual_Ex']<1) & (train['SalePrice']>13)].index)

#グラフを描画する
plt.figure(figsize=(20, 10))
plt.scatter(train["BsmtQual_Ex"],train["SalePrice"])
plt.xlabel("BsmtQual_Ex")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["YearRemodAdd"],train["SalePrice"])
plt.xlabel("YearRemodAdd")
plt.ylabel("SalePrice")
#外れ値を除外する
train = train.drop(train[(train['YearRemodAdd']<2000) & (train['SalePrice']>13)].index)

#グラフを描画する
plt.figure(figsize=(20, 10))
plt.scatter(train["YearRemodAdd"],train["SalePrice"])
plt.xlabel("YearRemodAdd")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["KitchenQual_Ex"],train["SalePrice"])
plt.xlabel("KitchenQual_Ex")
plt.ylabel("SalePrice")
plt.figure(figsize=(20, 10))
plt.scatter(train["Foundation_PConc"],train["SalePrice"])
plt.xlabel("Foundation_PConc")
plt.ylabel("SalePrice")
train_x = train[["1stFlrSF","TotalBsmtSF","Total_Bathrooms","TotalSF","YearBuilt","OverallQual","GrLivArea","GarageCars","FullBath","BsmtQual_Ex","YearRemodAdd","KitchenQual_Ex","Foundation_PConc"]]
train_y = train[["SalePrice"]]
test_x = test_x[["1stFlrSF","TotalBsmtSF","Total_Bathrooms","TotalSF","YearBuilt","OverallQual","GrLivArea","GarageCars","FullBath","BsmtQual_Ex","YearRemodAdd","KitchenQual_Ex","Foundation_PConc"]]
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(train_x.columns)
print(train_y.columns)
print(test_x.columns)
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
# データの分割
train_x, valid_x, train_y, valid_y = train_test_split(
        train_x,
        train_y,
        test_size=0.3,
        random_state=0)
dtrain = xgb.DMatrix(train_x, label=train_y)
dvalid = xgb.DMatrix(valid_x,label=valid_y)

num_round = 3000
evallist = [(dvalid, 'eval'), (dtrain, 'train')]

evals_result = {}

#パラメータ
param = {
            'max_depth': 3,
            'eta': 0.01,
            'objective': 'reg:squarederror',
}

#学習の実行
bst = xgb.train(
                        param, dtrain,
                        num_round,
                        evallist,
                        evals_result=evals_result,
                        # 一定ラウンド回しても改善が見込めない場合は学習を打ち切る
                        early_stopping_rounds=1000)
#学習曲線を可視化する
plt.figure(figsize=(20, 10))
train_metric = evals_result['train']['rmse']
plt.plot(train_metric, label='train rmse')
eval_metric = evals_result['eval']['rmse']
plt.plot(eval_metric, label='eval rmse')
plt.grid()
plt.legend()
plt.xlabel('rounds')
plt.ylabel('rmse')
plt.ylim(0, 0.3)
plt.show()
dtest = xgb.DMatrix(test_x)
submission = pd.DataFrame()
submission["Id"] = test_ID
submission["SalePrice"] = np.exp(bst.predict(dtest))
submission.to_csv('submission.csv', index=False)