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
#DS_K307_b 仮説2
#ライブラリ他インポート

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#データセットの読み込み

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
#データセット(訓練データ)の確認

train
#データセット(テストデータ)の確認

test
#データセット(提出フォーマット)の確認

sample_submission
#'SalePrice'のヒストグラムを確認

plt.figure(figsize = (4,3))

sns.distplot(train['SalePrice'])
#'SalePrice'のヒストグラムを確認　対数変換してみる

plt.figure(figsize = (4,3))

sns.distplot(np.log1p(train['SalePrice']))
#'SalePrice'を対数変換

train['SalePrice'] = np.log1p(train['SalePrice'])

train
#訓練データのカラムを表示

train.columns
#ひとまず、相関をとってみる

tc = train.corr()

tc
#'SalePrice'との相関を確認

tc1 = tc['SalePrice']

tc1
#仮説1の場合
#仮説１          そのまま   対数変換

#広さ

#1stFlrSF         0.605852 0.596981

#2ndFlrSF         0.319334 0.319300

#TotalBsmtSF      0.613581 0.612134



#築年数

#YearBuilt        0.522897 0.586570

#YearRemodAdd     0.507101 0.565608



#立地
#データを可視化してみる

#散布図を描いてみる

#'1stFlrSF'1Fの広さ

plt.figure(figsize = (4,3))

sns.scatterplot(x = '1stFlrSF', y = 'SalePrice',data = train)
#'2ndFlrSF'2Fの広さ

plt.figure(figsize = (4,3))

sns.scatterplot(x = '2ndFlrSF', y = 'SalePrice',data = train)
#'TotalBsmtSF'地下の広さ

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data = train)
#'YearBuilt'建築年

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'YearBuilt', y = 'SalePrice', data = train)
#'YearRemodAdd'改修年

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'YearRemodAdd', y = 'SalePrice', data = train)
#'1stFlrSF'と'2ndFlrSF'の相関を見てみる

tc_1st = tc['1stFlrSF']

tc_1st['2ndFlrSF']
#かなり弱い相関
#'1stFlrSF'と'TotalBsmtSF'の相関を見てみる

tc_1st['TotalBsmtSF']
#かなり強い正の相関がある
#'2ndFlrSF'と'TotalBsmtSF'の相関を見てみる

tc_2nd = tc['2ndFlrSF']

tc_2nd['TotalBsmtSF']
#かなり弱い相関
#新しい説明変数を追加

#延べ床面積を追加

train['TotalFloorArea'] = train['1stFlrSF'] + train['2ndFlrSF'] + train['TotalBsmtSF']

train
#'TotalFloorArea'延べ床面積

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'TotalFloorArea', y = 'SalePrice', data = train)
#'TotalFloorArea'と'SalePrice'の相関を見てみる

train.corr()['SalePrice']['TotalFloorArea']
#ここで、外れ値の削除を行うと標準化でエラーになる
#'YearBuilt'と'YearRemodAdd'の相関を見てみる

tc_year = tc['YearBuilt']

tc_year['YearRemodAdd']
#正の相関がある＞片方を削除してみる＞仮に'YearBuilt'を採用してみる
#多重比較法で立地('Neighborhood')の水準間に有意差があるかを確認

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)

res = pairwise_tukeyhsd(train['SalePrice'], groups = train['Neighborhood'], alpha = 0.01)

print(res.summary())
res.plot_simultaneous()
#多重比較法で立地('Condition1')の水準間に有意差があるかを確認

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)

res = pairwise_tukeyhsd(train['SalePrice'], groups = train['Condition1'], alpha = 0.01)

print(res.summary())
res.plot_simultaneous()
#多重比較法で立地('Condition2')の水準間に有意差があるかを確認

from statsmodels.stats.multicomp import (pairwise_tukeyhsd,MultiComparison)

res = pairwise_tukeyhsd(train['SalePrice'], groups = train['Condition2'], alpha = 0.01)

print(res.summary())
res.plot_simultaneous()
#Conditionの各水準間に有意差はほぼない
#仮説2の場合
#相関係数が>0.5のものを抽出

tc2 = tc1[abs(tc1) >= 0.5]

tc2
#相関係数が>0.5のもの 対数変換前

#OverallQual     0.790982

#YearBuilt       0.522897

#YearRemodAdd    0.507101

#TotalBsmtSF     0.613581

#1stFlrSF        0.605852

#GrLivArea       0.708624

#FullBath        0.560664

#TotRmsAbvGrd    0.533723

#GarageYrBlt     0.541073

#GarageCars      0.640409

#GarageArea      0.623431
#相関の確認

train[tc2.index].corr()
#データを可視化してみる

#散布図を描いてみる

#'OverallQual'家の全体的な素材と仕上げの評価

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'OverallQual', y = 'SalePrice', data = train)
#'YearBuilt'建築数

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'YearBuilt', y = 'SalePrice', data = train)
#'YearRemodAdd'改修年

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'YearRemodAdd', y = 'SalePrice', data = train)
#'TotalBsmtSF'地下の広さ

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'TotalBsmtSF', y = 'SalePrice', data = train)
#'1stFlrSF'1階の広さ

plt.figure(figsize = (4,3))

sns.scatterplot(x = '1stFlrSF', y = 'SalePrice', data = train)
#'GrLivArea'グレード（地上）のリビングエリアの平方フィート

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = train)
#'FullBath'グレードを超えるフルバスルーム

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'FullBath', y = 'SalePrice', data = train)
#'TotRmsAbvGrd'グレードを超える部屋の合計

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'TotRmsAbvGrd', y = 'SalePrice', data = train)
#'GarageYrBlt'ガレージが建てられた年

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'GarageYrBlt', y = 'SalePrice', data = train)
#'GarageCars'ガレージの大きさ(台数)

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'GarageCars', y = 'SalePrice', data = train)
#'GarageArea'ガレージの大きさ(面積)

plt.figure(figsize = (4,3))

sns.scatterplot(x = 'GarageArea', y = 'SalePrice', data = train)
#データの可視化ここまで
#'GarageCars'と'GarageArea'に相関がありそう。ここでは、GarageCarsを採用してみる
#'GarageYrBlt'と'YearBuilt'の相関を見てみる

tc_Gyear = tc['GarageYrBlt']

tc_Gyear['YearBuilt']
#かなり強い正の相関
#'YearRemodAdd'と'GarageYrBlt'の相関を見てみる

tc_yearR = tc['YearRemodAdd']

tc_yearR['GarageYrBlt']
#強い正の相関
#ここでは、'GarageYrBlt'を採用してみる
#テストデータに新しい説明変数'TotalFloorArea'を追加

test['TotalFloorArea'] = test['1stFlrSF'] + test['2ndFlrSF'] + test['TotalBsmtSF']

data = pd.concat([train,test], sort = False)

data
#目的変数'SalePrice'と'Id'削除

x = data.drop(['Id','SalePrice'], axis = 1)

x
#欠損の状況を確認

x_na = x.isnull().sum()[x.isnull().sum() > 0]

x_na
#欠損値の型を確認

x[x_na.index].dtypes
#説明変数を分類

#説明変数が数字のものだけを抽出

x_i_i = x.dtypes[x.dtypes == 'int64'].index

x_i = x[x_i_i]

x_i
#説明変数が少数を含むものだけを抽出

x_f_i = x.dtypes[x.dtypes == 'float64'].index

x_f = x[x_f_i]

x_f
#説明変数が文字のものだけを抽出

x_o_i = x.dtypes[x.dtypes == 'object'].index

x_o = x[x_o_i]

x_o
#少数を含む説明変数の欠損値を中央値で置き換える

#0で置き換える場合

#for i in x_f_i:

#    x_f.loc[x_f[i].isnull(),i] = 0

#x_f



for i in x_f_i:

    x_f[i].fillna(x_f[i].median(skipna = True), inplace = True)

x_f
#再度欠損状況を確認

x_f.isnull().sum()
#文字を含む説明変数の欠損値を文字"NA"で置き換える

for i in x_o_i:

    x_o.loc[x_o[i].isnull(),i] = 'NA'

x_o
#再度欠損状況を確認

x_o.isnull().sum()
#数字の説明変数を標準化

x_i_train = x_i[:len(train)]

x_i_test = x_i[len(train):]



from sklearn.preprocessing import StandardScaler

stds = StandardScaler()

x_i_train = pd.DataFrame(stds.fit_transform(x_i_train))

x_i_test = pd.DataFrame(stds.transform(x_i_test))

x_i[x_i.columns.tolist()] = pd.concat([x_i_train,x_i_test], sort = False)

x_i_s = x_i

x_i_s



#from sklearn.preprocessing import StandardScaler

#train_standard = StandardScaler()

#train_copied = x_i.copy()

#train_standard.fit(x_i)

#train_std = pd.DataFrame(train_standard.transform(x_i))

#x_i[x_i.columns.tolist()] = train_std

#x_i_s = x_i

#x_i_s
#少数を含む数字の説明変数を標準化

x_f_train = x_f[:len(train)]

x_f_test = x_f[len(train):]



from sklearn.preprocessing import StandardScaler

stds = StandardScaler()

x_f_train = pd.DataFrame(stds.fit_transform(x_f_train))

x_f_test = pd.DataFrame(stds.transform(x_f_test))

x_f[x_f.columns.tolist()] = pd.concat([x_f_train,x_f_test], sort = False)

x_f_s = x_f

x_f_s



#from sklearn.preprocessing import StandardScaler

#train_standard = StandardScaler()

#train_copied = x_f.copy()

#train_standard.fit(x_f)

#train_std = pd.DataFrame(train_standard.transform(x_f))

#x_f[x_f.columns.tolist()] = train_std

#x_f_s = x_f

#x_f_s
#文字を含む説明変数をダミー変数化

x_o_neighbor = x_o['Neighborhood']#仮説1

x_o_d = pd.get_dummies(x_o, drop_first = True)#仮説2、仮説1：_neighbor_neighbor

x_o_d
#数字の説明変数を結合

x_n = pd.concat([x_i_s,x_f_s], axis=1, sort = False)

x_n
#使用する説明変数を抽出してカテゴリ変数を追加

x_n_c1 = ["OverallQual","GrLivArea","GarageArea","GarageCars","1stFlrSF","TotalBsmtSF"]

x_n_c2 = ["OverallQual","GrLivArea","GarageCars","TotalBsmtSF"]

x_n_c3 = ["2ndFlrSF","TotalBsmtSF","YearBuilt"]

x_n_c4 = ["TotalFloorArea","YearBuilt"]

x_n_c5 = ["TotalFloorArea","YearBuilt","OverallQual","GrLivArea","GarageCars"]

x_n_c6 = ["GarageCars","TotalFloorArea","GarageYrBlt","OverallQual","GrLivArea","FullBath","TotRmsAbvGrd"]#仮説2

x_final = pd.concat([x_n[x_n_c6],x_o_d], axis = 1, sort = False)#仮説2

x_final
#訓練データとテストデータを分割

x_train = x_final[:len(train)]

x_train
#データ分割の確認(テストデータ(ホールドアウト))

x_test = x_final[len(train):]

x_test
#正解確認

y_train = train['SalePrice']

y_train
#訓練データを分割し80%を学習データに、20%をテストデータにする(ホールドアウト)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)
#学習する

#線形回帰

#from sklearn.linear_model import LinearRegression

#model = LinearRegression()



#ロジスティック回帰

#from sklearn.linear_model import LogisticRegression

#model = LogisticRegression()



#Lasso回帰

from sklearn.linear_model import Lasso

model = Lasso(alpha = 0.001, max_iter = 1000000)



#Ridge回帰

#from sklearn.linear_model import Ridge

#model = Ridge(alpha=50, max_iter=1000000)



model.fit(X_train, Y_train)
#係数の確認

model.coef_
#Lasso回帰では、いくつかの係数がゼロになっている
#訓練データを予想

train_predicted = model.predict(X_train)

train_predicted
#正解確認

Y_train
#MSEを確認(訓練データ)

from sklearn.metrics import mean_squared_error

mean_squared_error(Y_train, train_predicted)
#テストデータ(ホールドアウト)を予想

test_predicted = model.predict(X_test)

test_predicted
#正解確認

Y_test
#MSEを確認(テストデータ(ホールドアウト))

mean_squared_error(Y_test, test_predicted)
#精度の表示

print(f"学習データの予測精度: {model.score(X_train, Y_train):.2}")

print(f"テストデータの予測精度: {model.score(X_test, Y_test):.2f}")
#提出用テストデータを予測

test_predicted2 = model.predict(x_test)

test_predicted2
#提出用ファイルの作成

sample_submission['SalePrice'] = list(map(int, np.exp(test_predicted2)))

sample_submission.to_csv('submission.csv', index = False)

sample_submission
#プログラムここまで