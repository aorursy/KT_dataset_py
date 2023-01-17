import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns

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
# データの読み込み

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv') #訓練データ

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv') #テストデータ
# 訓練/テストデータのサイズ確認 (行,列)

print('The size of train is : ' + str(train.shape))

print('The size of test is : ' + str(test.shape))
# 訓練データ

train.head()
# テストデータ

# SalePriceが存在しない

test.head()
# 学習データとテストデータをまとめて前処理するためにマージを行う

# WhatIsData: 訓練/テストデータであることを区別する列を追加

train['WhatIsData'] = 'Train'

test['WhatIsData'] = 'Test'

# テストデータに存在しないSalesPriceを'9999999999'で追加

test['SalePrice'] = 9999999999

# alldataに訓練/テストデータを集約

alldata = pd.concat([train,test],axis=0).reset_index(drop=True)
# マージ処理後のデータサイズを確認

print('The size of train is : ' + str(train.shape))

print('The size of test is : ' + str(test.shape))

print('The size of alldata is : ' + str(alldata.shape))
alldata.head()
# 学習データの欠損状況

train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
# テストデータの欠損状況

test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
# 欠損を含むカラムのデータ型を確認

na_col_list = alldata.isnull().sum()[alldata.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

alldata[na_col_list].dtypes.sort_values() #データ型
# alldataに対して欠損値の補完

# floatの場合は0

# objectの場合は'NA'

na_float_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='float64'].index.tolist() #float64

na_obj_cols = alldata[na_col_list].dtypes[alldata[na_col_list].dtypes=='object'].index.tolist() #object

# float64型で欠損している場合は0を代入

for na_float_col in na_float_cols:

    alldata.loc[alldata[na_float_col].isnull(),na_float_col] = 0.0

# object型で欠損している場合は'NA'を代入

for na_obj_col in na_obj_cols:

    alldata.loc[alldata[na_obj_col].isnull(),na_obj_col] = 'NA'
# alldataの欠損値が補完されていることを確認

# マージデータの欠損状況

alldata.isnull().sum()[alldata.isnull().sum()>0].sort_values(ascending=False)
# カテゴリカル変数の特徴量をリスト化

cat_cols = alldata.dtypes[alldata.dtypes=='object'].index.tolist()

# 数値変数の特徴量をリスト化

num_cols = alldata.dtypes[alldata.dtypes!='object'].index.tolist()

# データ分割および提出時に必要なカラムをリスト化

other_cols = ['Id','WhatIsData']

# 余計な要素をリストから削除

cat_cols.remove('WhatIsData') #学習データ・テストデータ区別フラグ除去

num_cols.remove('Id') #Id削除

# カテゴリカル変数をダミー化

alldata_cat = pd.get_dummies(alldata[cat_cols])

# データ統合

all_data = pd.concat([alldata[other_cols],alldata[num_cols],alldata_cat],axis=1)
# sns.distplotを使ったヒストグラムの確認

sns.distplot(train['SalePrice'])
# sns.distplotを使ったヒストグラムの確認

sns.distplot(np.log(train['SalePrice']))
# マージデータから学習データを取得

train_ = all_data[all_data['WhatIsData']=='Train'].drop(['WhatIsData','Id'], axis=1).reset_index(drop=True)

# 　訓練データを説明変数と目的変数に分割

train_x = train_.drop('SalePrice',axis=1)

train_y = np.log(train_['SalePrice'])
# グリッドを用いて最良のパラメータを検索する

scaler = StandardScaler()  #スケーリング

param_grid = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] #パラメータグリッド

cnt = 0

for alpha in param_grid:

    #Lasso回帰モデル

    ls = Lasso(alpha=alpha)

    #パイプライン生成

    pipeline = make_pipeline(scaler, ls)

    # 訓練データを使ってモデル構築用のデータを生成

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    # 予測値の算出

    py_test  = pipeline.predict(X_test)

    # RMSE:二乗平均平方根誤差(スコア)の算出

    test_rmse = np.sqrt(mean_squared_error(y_test, py_test))

    # 初回実行時の処理

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    # 最小となるbest_scoreを求める

    elif best_score > test_rmse:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    else:

        pass

    cnt = cnt + 1

    

print('alpha : ' + str(best_param))

print('test score is : ' +str(best_score))
# グリッドを用いて最良のパラメータを検索する

scaler = StandardScaler()  #スケーリング

param_grid = [0.001, 0.01, 0.1, 1.0, 10.0,100.0,1000.0] #パラメータグリッド

cnt = 0

for alpha in param_grid:

    #Lasso回帰モデル

    ls = Lasso(alpha=alpha)

    #パイプライン生成

    pipeline = make_pipeline(scaler, ls)

    # 訓練データを使ってモデル構築用のデータを生成

    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

    pipeline.fit(X_train,y_train)

    # 予測値の算出

    py_test  = pipeline.predict(X_test)

    # RMSE:二乗平均平方根誤差(スコア)の算出

    test_rmse = np.sqrt(mean_squared_error(y_test, py_test))

    # 初回実行時の処理

    if cnt == 0:

        best_score = test_rmse

        best_estimator = pipeline

        best_param = alpha

    # 最小となるbest_scoreを求める

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
# マージデータからテストデータを取得

test_ = all_data[all_data['WhatIsData']=='Test'].drop(['WhatIsData','SalePrice'], axis=1).reset_index(drop=True)

# テストデータを説明変数と目的変数に分割

test_id = test_['Id']

test_data = test_.drop('Id',axis=1)
# 提出用データ生成

# test_id

ls = Lasso(alpha = 0.01)

pipeline = make_pipeline(scaler, ls)

pipeline.fit(train_x,train_y)

test_SalePrice = pd.DataFrame(np.exp(pipeline.predict(test_data)),columns=['SalePrice'])

test_Id = pd.DataFrame(test_id,columns=['Id'])

pd.concat([test_Id, test_SalePrice],axis=1).to_csv('output.csv',index=False)