# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



#scikit-learn

from sklearn import tree

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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
#データの確認

def all_data(df):

    display(df.head().append(df.tail()))

    

print('train data')

all_data(df1)

print('test data')

all_data(df2)
train_x = df1.drop('SalePrice', axis=1)

train_y = df1['SalePrice']

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
#相関係数のヒートマップ

corrmat = df1.corr()

corr = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

cc = np.corrcoef(df1[corr].values.T)

fig, ax = plt.subplots(figsize=(15,15))

sns.set(font_scale=2.2)

sns.heatmap(cc, cbar=True, annot=True, fmt='2f', annot_kws={'size': 10}, yticklabels=corr.values, xticklabels=corr.values)

plt.show()

fig.savefig('figure4.png')
#説明変数の候補となる６つのデータ確認

#train data

ID = df1['Id']

OverallQual = df1['OverallQual']

GrLivArea = df1['GrLivArea']

GarageCars = df1['GarageCars']

TotalBsmtSF = df1['TotalBsmtSF']

FullBath = df1['FullBath']

YearBuilt = df1['YearBuilt']

#test data

ID_2 = df2['Id']

OverallQual_2 = df2['OverallQual']

GrLivArea_2 = df2['GrLivArea']

GarageCars_2 = df2['GarageCars']

TotalBsmtSF_2 = df2['TotalBsmtSF']

FullBath_2 = df2['FullBath']

YearBuilt_2 = df2['YearBuilt']
#train data

df3 = pd.concat([ID, OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt], axis=1)

df3
df4 = pd.concat([ID_2, OverallQual_2, GrLivArea_2, GarageCars_2, TotalBsmtSF_2, FullBath_2, YearBuilt_2], axis=1)

df4
#欠損値の確認

def nan_data(df):

    display(df.isnull().sum())



print('train data')

nan_data(df3)

print('test data')

nan_data(df4)
#GarageCarsの欠損値に0を代入

df2['GarageCars'] = df2['GarageCars'].fillna(0)

#TotalBsmtSFの欠損値に0を代入

df2['TotalBsmtSF'] = df2['TotalBsmtSF'].fillna(0)
#欠損値の確認

GarageCars_2 = df2['GarageCars']

TotalBsmtSF_2 = df2['TotalBsmtSF']

df4 = pd.concat([ID_2, OverallQual_2, GrLivArea_2, GarageCars_2, TotalBsmtSF_2, FullBath_2, YearBuilt_2], axis=1)

df4.isnull().sum()
#SalePriceの統計量

df1['SalePrice'].describe()
#説明変数の統計量

def data_des(df):

    display(df.describe())

    

print('train data')

data_des(df3)

print('test data')

data_des(df4)
sns.distplot(df1['SalePrice'])
#歪度、尖度

print("歪度: %f" % df1['SalePrice'].skew())

print("尖度: %f" % df1['SalePrice'].kurt())
#正規分布に近づける

sns.distplot(np.log(df1['SalePrice']))
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
plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x='GrLivArea',y='SalePrice',data=df1)
#外れ値をGrLivArea>4000とする。

df1 = df1.drop(df1[(df1['GrLivArea']>4000)].index)

plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x='GrLivArea',y='SalePrice',data=df1)
fig, ax = plt.subplots(figsize=(15,15))

x=df1['GarageCars'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('GarageCars', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('GarageCars', fontsize=15)

plt.show()
#箱ひげ図

fig, ax = plt.subplots(figsize=(15,15))

ax.boxplot([df1['TotalBsmtSF'], df2['TotalBsmtSF']], labels=['train data', 'test data'])

ax.set_title('TotalBsmtSF')

ax.set_ylabel('Data')

plt.show()
#散布図

fig, ax = plt.subplots(figsize=(15,15))

x=df1['TotalBsmtSF'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('TotalBsmtSF', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('TotalBsmtSF', fontsize=15)

plt.show()
df1 = df1.drop(df1[(df1['TotalBsmtSF'] > 3000) & (df1['SalePrice'] < 300000)].index)

fig, ax = plt.subplots(figsize=(15,15))

x=df1['TotalBsmtSF'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('TotalBsmtSF', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('TotalBsmtSF', fontsize=15)

plt.show()
fig, ax = plt.subplots(figsize=(15,15))

x=df1['FullBath'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('FullBath', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('FullBath', fontsize=15)

plt.show()
#外れ値をFullBath=0,SalePrice>300000とする。

df1 = df1.drop(df1[(df1['FullBath'] == 0) & (df1['SalePrice'] > 300000)].index)

fig, ax = plt.subplots(figsize=(15,15))

x=df1['FullBath'].values

y=df1['SalePrice'].values

plt.scatter(x, y)

plt.xlabel('FullBath', fontsize=15)

plt.ylabel('SalePrice', fontsize=15)

plt.title('FullBath', fontsize=15)

plt.show()
plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x='YearBuilt',y='SalePrice',data=df1)
#外れ値をYearBuilt<2000,SalePrice>600000とする。

df1 = df1.drop(df1[(df1['YearBuilt']<2000) & (df1['SalePrice']>600000)].index)

plt.figure(figsize=(20, 10))

plt.xticks(rotation='90')

sns.boxplot(x='YearBuilt',y='SalePrice',data=df1)
#train data

ID = df1['Id']

OverallQual = df1['OverallQual']

GrLivArea = df1['GrLivArea']

GarageCars = df1['GarageCars']

TotalBsmtSF = df1['TotalBsmtSF']

FullBath = df1['FullBath']

YearBuilt = df1['YearBuilt']

df3 = pd.concat([ID, OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt], axis=1)

#test data

ID_2 = df2['Id']

OverallQual_2 = df2['OverallQual']

GrLivArea_2 = df2['GrLivArea']

GarageCars_2 = df2['GarageCars']

TotalBsmtSF_2 = df2['TotalBsmtSF']

FullBath_2 = df2['FullBath']

YearBuilt_2 = df2['YearBuilt']

df4 = pd.concat([ID_2, OverallQual_2, GrLivArea_2, GarageCars_2, TotalBsmtSF_2, FullBath_2, YearBuilt_2], axis=1)
#train data, test data, 目的変数の3つに分ける

train_data = df3.values

test_data = df4.values

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
solution.to_csv('tree1.csv', index_label = ['Id'])
#予測データの確認

print(solution)
#train dataと予測データの比較

sns.distplot(df1['SalePrice'])

plt.show()

sns.distplot(solution)

plt.show()
train_X = df3.drop(['Id'], axis=1)

train_y = df1['SalePrice']

from sklearn.ensemble import RandomForestRegressor

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

X = df3.drop(['Id'], axis=1)

Y = df1['SalePrice']



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=60)



tree_1 = DecisionTreeRegressor(max_depth=6, random_state=60)

tree_1 = tree_1.fit(X_train, Y_train)



print('train:', tree_1.score(X_train, Y_train))

print('test:', tree_1.score(X_test, Y_test))
#train data

ID = df1['Id']

OverallQual = df1['OverallQual']

GrLivArea = df1['GrLivArea']

TotalBsmtSF = df1['TotalBsmtSF']

YearBuilt = df1['YearBuilt']

GarageCars = df1['GarageCars']

df5 = pd.concat([ID, OverallQual, GrLivArea, TotalBsmtSF, YearBuilt, GarageCars], axis=1)

#test data

ID_2 = df2['Id']

OverallQual_2 = df2['OverallQual']

GrLivArea_2 = df2['GrLivArea']

TotalBsmtSF_2 = df2['TotalBsmtSF']

YearBuilt_2 = df2['YearBuilt']

GarageCars_2 = df2['GarageCars']

df6 = pd.concat([ID_2, OverallQual_2, GrLivArea_2, TotalBsmtSF_2, YearBuilt_2, GarageCars_2], axis=1)
#train data, test data, 目的変数の3つに分ける

train_data = df5.values

test_data = df6.values

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
solution_1.to_csv('tree2.csv', index_label = ['Id'])
#予測データの確認

print(solution_1)
#train dataと予測データの比較

sns.distplot(df1['SalePrice'])

plt.show()

sns.distplot(solution_1)

plt.show()
train_x = df5

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

x = df5

y = df1['SalePrice']



x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=60)



tree_2 = DecisionTreeRegressor(max_depth=6, random_state=60)

tree_2 = tree_2.fit(x_train, y_train)



print('train:', tree_2.score(x_train, y_train))

print('test:', tree_2.score(x_test, y_test))