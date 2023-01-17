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
train.describe
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
import seaborn as sns

# ヒートマップのフォントサイズを指定

sns.set(font_scale=1.25)



# 算出した相関データをヒートマップで表示

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
# 散布図の表示

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea']

sns.pairplot(train[cols], size = 2.5)
# 学習データの欠損状況

# 欠損値を含むカラムについて欠損値の数を数が多い順に表示

train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
# 欠損を含むカラムのデータ型を確認

na_col_list = train.isnull().sum()[train.isnull().sum()>0].index.tolist()

train[na_col_list].dtypes.sort_values()
# データ型に応じて欠損値を補完

# floatの場合は0

# objectの場合は'NA'

na_float_cols = train[na_col_list].dtypes[train[na_col_list].dtypes=='float64'].index.tolist()

na_obj_cols = train[na_col_list].dtypes[train[na_col_list].dtypes=='object'].index.tolist()



# float64型で欠損している場合は0を代入

for na_float_col in na_float_cols:

    train.loc[train[na_float_col].isnull(),na_float_col] = 0.0

    

# object型で欠損している場合は'NA'を代入

for na_obj_col in na_obj_cols:

    train.loc[train[na_obj_col].isnull(),na_obj_col] = 'NA'
# マージデータの欠損状況

train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
train = train.drop(train[(train['OverallQual']>=10) & (train['SalePrice']<250000)].index)
train = train.drop(train[(train['YearBuilt']>=2000) & (train['SalePrice']<100000)].index)

train = train.drop(train[(train['YearBuilt']<=1900) & (train['SalePrice']>400000)].index)
train = train.drop(train[(train["GrLivArea"]>=3000) & (train['SalePrice']<300000)].index)
# カテゴリカル変数の特徴量をリスト化

cat_cols = train.dtypes[train.dtypes=='object'].index.tolist()

# 数値変数の特徴量をリスト化

num_cols = train.dtypes[train.dtypes!='object'].index.tolist()

# データ分割および提出時に必要なカラムをリスト化

other_cols = ['Id']

# 余計な要素をリストから削除

num_cols.remove('Id') #Id削除

# カテゴリカル変数をダミー化

train_cat = pd.get_dummies(train[cat_cols])

# データ統合

train = pd.concat([train[other_cols],train[num_cols],train_cat],axis=1)
import statsmodels.api as smf
# 説明変数と目的変数に分ける

# 説明変数にはSalePriceと高い相関が見られたもののみを採用

aiu = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

x_train = train[aiu]

y_train = train['SalePrice']
model = smf.OLS(y_train, x_train)   # モデルの設定

result = model.fit()   # 回帰分析の実行
# 結果を表示

# Adj. R-squared (uncentered) の値が１に近いほど分析の精度が高い

print(result.summary())
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# テストデータの欠損状況

# 欠損値を含むカラムについて欠損値の数を数が多い順に表示

test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
# 欠損を含むカラムのデータ型を確認

na_col_list = test.isnull().sum()[test.isnull().sum()>0].index.tolist()

test[na_col_list].dtypes.sort_values()
# データ型に応じて欠損値を補完

# floatの場合は0

# objectの場合は'NA'

na_float_cols = test[na_col_list].dtypes[test[na_col_list].dtypes=='float64'].index.tolist()

na_obj_cols = test[na_col_list].dtypes[test[na_col_list].dtypes=='object'].index.tolist()



# float64型で欠損している場合は0を代入

for na_float_col in na_float_cols:

    test.loc[test[na_float_col].isnull(),na_float_col] = 0.0

    

# object型で欠損している場合は'NA'を代入

for na_obj_col in na_obj_cols:

    test.loc[test[na_obj_col].isnull(),na_obj_col] = 'NA'
# マージデータの欠損状況

test.isnull().sum()[test.isnull().sum()>0].sort_values(ascending=False)
# カテゴリカル変数の特徴量をリスト化

cat_cols = test.dtypes[test.dtypes=='object'].index.tolist()

# 数値変数の特徴量をリスト化

num_cols = test.dtypes[test.dtypes!='object'].index.tolist()

# データ分割および提出時に必要なカラムをリスト化

other_cols = ['Id']

# 余計な要素をリストから削除

num_cols.remove('Id') #Id削除

# カテゴリカル変数をダミー化

test_cat = pd.get_dummies(test[cat_cols])

# データ統合

test = pd.concat([test[other_cols],test[num_cols],test_cat],axis=1)
# 説明変数を選択

aiu = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

x_test = test[aiu]



predictions = result.predict(x_test)

print(predictions)
test["SalePrice"] = predictions
test[["Id","SalePrice"]].head()
# Id, SalePriceの2列だけのファイルに変換

test[["Id","SalePrice"]].to_csv("submission.csv",index=False)