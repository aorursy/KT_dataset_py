# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline

from sklearn import tree

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#データのインポート

df1 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df2 = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
#データの大きさ

def data_size(df):

    display(str(df.shape))

    

print('train data')

data_size(df1)

print('test data')

data_size(df2)
#変数の確認

def keys_data(df):

    display(df.keys())

    

print('train data')

keys_data(df1)

print('test data')

keys_data(df2)
#変数の型の確認

def data_type(df):

    display(df.dtypes)

    

print('train data')

data_type(df1)

print('test data')

data_type(df2)
#train data, test dataの確認

def all_data(df):

    display(df.head().append(df.tail()))

    

print('train data')

all_data(df1)

print('test data')

all_data(df2)
#SalePriceの統計量

df1['SalePrice'].describe()
#SalePriceのヒストグラム

sns.distplot(df1['SalePrice'])
#歪度、尖度

print("歪度: %f" % df1['SalePrice'].skew())

print("尖度: %f" % df1['SalePrice'].kurt())
#正規分布に近づける

sns.distplot(np.log(df1['SalePrice']))
#train dataを目的変数とそれ以外に分ける。

train_x = df1.drop('SalePrice', axis=1)

train_y = df1['SalePrice'].values

#train dataとtest dataを結合

all_data = pd.concat([train_x,df2],axis=0,sort=True)
#それぞれのデータのサイズを確認

print('train_x: '+str(train_x.shape)) #説明変数のサイズ

print('train_y: '+str(train_y.shape)) #目的変数のサイズ

print('all_data: '+str(all_data.shape)) #train dataとtest dataの説明変数のサイズの合計
#欠損値の確認

nan_data = all_data.isnull().sum()[all_data.isnull().sum()>0].sort_values(ascending=False)

nan_data
#欠損値の数のグラフ化

plt.figure(figsize=(20,15))

plt.xticks(rotation='90')

sns.barplot(x=nan_data.index, y=nan_data)
# 欠損を含むカラムのデータ型を確認

na_col_list = all_data.isnull().sum()[all_data.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化

all_data[na_col_list].dtypes.sort_values() #データ型
#feature importance

def random_frg(df):

    train_x = df.drop(['SalePrice'], axis=1)

    train_y = df['SalePrice']

    rf = RandomForestRegressor()

    rf.fit(train_x, train_y)

    ranking = np.argsort(rf.feature_importances_)

    fig, ax = plt.subplots(figsize=(20,10))

    plt.barh(range(len(ranking)), rf.feature_importances_[ranking], color='r')

    plt.yticks(range(len(ranking)), train_x.columns[ranking])

    ax.set_xlabel('feature importance')

    ax.set_title('Feature Importance')

    plt.tight_layout()

    plt.show()

    fig.savefig('figure.png')
#float64型の変数には0を、object型の変数にはNAを欠損に代入

for col in df1:

    if df1[col].dtype !='object':

        df1[col].fillna(0.0, inplace=True)

    else:

        df1[col].fillna('NA', inplace=True)

df1 = df1._get_numeric_data()

random_frg(df1)
#train data 

#説明変数

OverallQual = df1['OverallQual']

GrLivArea = df1['GrLivArea']

TotalBsmtSF= df1['TotalBsmtSF']

FlrSF1st = df1['1stFlrSF']

FlrSF2nd = df2['2ndFlrSF']

BsmtFinSF1 = df1['BsmtFinSF1']

GarageCars = df1['GarageCars']

LotArea = df1['LotArea']

GarageArea = df1['GarageArea']

YearBuilt = df1['YearBuilt']

YearRemodAdd = df1['YearRemodAdd']

GarageYrBlt = df1['GarageYrBlt']

TotRmsAbvGrd = df1['TotRmsAbvGrd']

LotFrontage = df1['LotFrontage']

BsmtUnfSF = df1['BsmtUnfSF']

WoodDeckSF = df1['WoodDeckSF']

OpenPorchSF = df1['OpenPorchSF']

OverallCond = df1['OverallCond']

FullBath = df1['FullBath']

#目的変数

SalePrice = df1['SalePrice']
#データの抽出

df3 = pd.concat([OverallQual, GrLivArea, TotalBsmtSF, FlrSF1st, FlrSF2nd, BsmtFinSF1, GarageCars, LotArea, GarageArea, YearBuilt, YearRemodAdd,  GarageYrBlt, TotRmsAbvGrd, LotFrontage, BsmtUnfSF, WoodDeckSF, OpenPorchSF, OverallCond, FullBath, SalePrice], axis=1)

df3
#salePriceとの相関を表すヒートマップ

corrmat = df3.corr()

corr = corrmat.nlargest(18, 'SalePrice')['SalePrice'].index

cc = np.corrcoef(df1[corr].values.T)

fig, ax = plt.subplots(figsize=(15,15))

sns.set(font_scale=2.2)

sns.heatmap(cc, cbar=True, annot=True, fmt='2f', annot_kws={'size': 8}, yticklabels=corr.values, xticklabels=corr.values)

plt.show()

fig.savefig('figure3.png')
#説明変数(train)

OverallQual = df1['OverallQual']

GrLivArea = df1['GrLivArea']

FlrSF2nd = df1['2ndFlrSF']

GarageArea = df1['GarageArea']

TotalBsmtSF= df1['TotalBsmtSF']

YearBuilt = df1['YearBuilt']

YearRemodAdd = df1['YearRemodAdd']

#説明変数(test)

OverallQual_2 = df2['OverallQual']

GrLivArea_2 = df2['GrLivArea']

TotalBsmtSF_2 = df2['TotalBsmtSF']

FlrSF2nd_2 = df2['2ndFlrSF']

GarageArea_2 = df2['GarageArea']

YearBuilt_2 = df2['YearBuilt']

YearRemodAdd_2 = df2['YearRemodAdd']

#目的変数

SalePrice = df1['SalePrice']
#データの抽出(train)

df4 = pd.concat([OverallQual, GrLivArea, FlrSF2nd, GarageArea, TotalBsmtSF, YearBuilt, YearRemodAdd, SalePrice], axis=1)

df4
#データの抽出(test)

df5 = pd.concat([OverallQual_2, GrLivArea_2, FlrSF2nd_2, GarageArea_2, TotalBsmtSF_2, YearBuilt_2, YearRemodAdd_2], axis=1)

df5
#欠損の確認(train)

df4.isnull().sum()
#欠損の確認(test)

df5.isnull().sum()
df2['GarageArea'] = df2['GarageArea'].fillna(0)

df2['TotalBsmtSF'] = df2['TotalBsmtSF'].fillna(0)
df5 = pd.concat([OverallQual_2, GrLivArea_2, FlrSF2nd_2, GarageArea_2, TotalBsmtSF_2, YearBuilt_2, YearRemodAdd_2], axis=1)

df5.isnull().sum()
#説明変数の統計量

df4.describe()
fig, ax = plt.subplots(figsize=(15,15))

x=df1['OverallQual'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('OverallQual', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('OverallQual', fontsize=15)

plt.show()
#外れ値がOverallQual=10,SalePrice<200000とする。

df1 = df1.drop(df1[(df1['OverallQual'] == 10) & (df1['SalePrice'] < 200000)].index)

fig, ax = plt.subplots(figsize=(15,15))

x=df1['OverallQual'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('OverallQual', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('OverallQual', fontsize=15)

plt.show()
sns.distplot(df1['2ndFlrSF'])
fig, ax = plt.subplots(figsize=(15,15))

x=df1['2ndFlrSF'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('2ndFlrSF', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('2ndFlrSF', fontsize=15)

plt.show()
sns.distplot(df1['GarageArea'])
fig, ax = plt.subplots(figsize=(15,15))

x=df1['GarageArea'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('GarageArea', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('GarageArea', fontsize=15)

plt.show()
#GarageArea>1200, SalePrice<300000を外れ値と考える

df1 = df1.drop(df1[(df1['GarageArea'] > 1200) & (df1['SalePrice'] < 300000)].index)

fig, ax = plt.subplots(figsize=(15,15))

x=df1['GarageArea'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('GarageArea', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('GarageArea', fontsize=15)

plt.show()
sns.distplot(df1['TotalBsmtSF'])
fig, ax = plt.subplots(figsize=(15,15))

x=df1['TotalBsmtSF'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('TotalBsmtSF', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('TotalBsmtSF', fontsize=15)

plt.show()
plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x='YearBuilt',y='SalePrice',data=df1)
#YearBuilt<2000,SalePrice>600000を外れ値と考える。

df1 = df1.drop(df1[(df1['YearBuilt']<2000) & (df1['SalePrice']>600000)].index)

plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x='YearBuilt',y='SalePrice',data=df1)
plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x='YearRemodAdd',y='SalePrice',data=df1)
#説明変数(train)

OverallQual = df1['OverallQual']

GrLivArea = df1['GrLivArea']

FlrSF2nd = df1['2ndFlrSF']

GarageArea = df1['GarageArea']

TotalBsmtSF= df1['TotalBsmtSF']

YearBuilt = df1['YearBuilt']

YearRemodAdd = df1['YearRemodAdd']

df4 = pd.concat([OverallQual, GrLivArea, FlrSF2nd, GarageArea, TotalBsmtSF, YearBuilt, YearRemodAdd], axis=1)

#説明変数(test)

OverallQual_2 = df2['OverallQual']

GrLivArea_2 = df2['GrLivArea']

TotalBsmtSF_2 = df2['TotalBsmtSF']

FlrSF2nd_2 = df2['2ndFlrSF']

GarageArea_2 = df2['GarageArea']

YearBuilt_2 = df2['YearBuilt']

YearRemodAdd_2 = df2['YearRemodAdd']

df5 = pd.concat([OverallQual_2, GrLivArea_2, FlrSF2nd_2, GarageArea_2, TotalBsmtSF_2, YearBuilt_2, YearRemodAdd_2], axis=1)
#train data, test data, 目的変数の3つに分ける

train_data = df4.values

test_data = df5.values

target = df1['SalePrice'].values
#決定木系のモデルを利用

tree_1 = tree.DecisionTreeClassifier()

tree_1 = tree_1.fit(train_data, target)
#test dataの説明変数を利用して予測

prediction_1 = tree_1.predict(test_data)

#予測データのサイズ確認

prediction_1.shape
#予測データの中身

print(prediction_1)
ID_0 = np.array(df2['Id']).astype(int)

solution = pd.DataFrame(prediction_1, ID_0, columns = ['SalePrice'])
solution.to_csv('tree3.csv', index_label = ['Id'])
#予測データの確認

print(solution)
#train dataと予測データの比較

sns.distplot(df1['SalePrice'])

plt.show()

sns.distplot(solution)

plt.show()
train_X = df4

train_y = df1['SalePrice']

rf = RandomForestRegressor()

#feature importanceで特徴量の重要度を確認

tree_1 = tree.DecisionTreeClassifier(random_state=0)

tree_1 = tree_1.fit(train_X, train_y)

rank = np.argsort(-tree_1.feature_importances_)

fig, ax = plt.subplots(figsize=(15,15))

sns.barplot(x=tree_1.feature_importances_[rank], y=train_X.columns.values[rank], orient='h')

plt.title('Feature Importance')

plt.tight_layout()

plt.show()
#予測結果算出

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

X = df4

Y = df1['SalePrice']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=60)



tree_1 = DecisionTreeRegressor(max_depth=6, random_state=60)

tree_1 = tree_1.fit(X_train, Y_train)



print('train:', tree_1.score(X_train, Y_train))

print('test:', tree_1.score(X_test, Y_test))
#説明変数(train)

GrLivArea = df1['GrLivArea']

GarageArea = df1['GarageArea']

TotalBsmtSF= df1['TotalBsmtSF']

YearBuilt = df1['YearBuilt']

YearRemodAdd = df1['YearRemodAdd']

FlrSF2nd = df1['2ndFlrSF']

df6 = pd.concat([GrLivArea, GarageArea, TotalBsmtSF, YearBuilt, YearRemodAdd, FlrSF2nd], axis=1)

#説明変数(test)

GrLivArea_2 = df2['GrLivArea']

TotalBsmtSF_2 = df2['TotalBsmtSF']

GarageArea_2 = df2['GarageArea']

YearBuilt_2 = df2['YearBuilt']

YearRemodAdd_2 = df2['YearRemodAdd']

FlrSF2nd_2 = df2['2ndFlrSF']

df7 = pd.concat([GrLivArea_2, GarageArea_2, TotalBsmtSF_2, YearBuilt_2, YearRemodAdd_2, FlrSF2nd_2], axis=1)
#train data, test data, 目的変数の3つに分ける

train_data = df6.values

test_data = df7.values

target = df1['SalePrice'].values
#決定木系のモデルを利用

tree_2 = tree.DecisionTreeClassifier()

tree_2 = tree_1.fit(train_data, target)
#test dataの説明変数を利用して予測

prediction_2 = tree_2.predict(test_data)

#予測データのサイズ確認

prediction_2.shape
#予測データの中身

print(prediction_2)
ID_1 = np.array(df2['Id']).astype(int)

solution_1 = pd.DataFrame(prediction_2, ID_1, columns = ['SalePrice'])

solution_1.to_csv('tree4.csv', index_label = ['Id'])
#予測データの確認

print(solution_1)
#train dataと予測データの比較

sns.distplot(df1['SalePrice'])

plt.show()

sns.distplot(solution_1)

plt.show()
train_x = df6

train_y = df1['SalePrice']

rf = RandomForestRegressor()

#feature importanceで特徴量の重要度を確認

tree_2 = tree.DecisionTreeClassifier(random_state=0)

tree_2 = tree_2.fit(train_x, train_y)

rank = np.argsort(-tree_2.feature_importances_)

fig, ax = plt.subplots(figsize=(15,15))

sns.barplot(x=tree_2.feature_importances_[rank], y=train_x.columns.values[rank], orient='h')

plt.title('Feature Importance')

plt.tight_layout()

plt.show()
#予測結果算出

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

x = df6

y = df1['SalePrice']



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=60)



tree_2 = DecisionTreeRegressor(max_depth=6, random_state=60)

tree_2 = tree_2.fit(x_train, y_train)



print('train:', tree_2.score(x_train, y_train))

print('test:', tree_2.score(x_test, y_test))