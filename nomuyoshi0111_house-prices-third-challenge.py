#invite people for the Kaggle party

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()
train_ids = df_train['Id']

test_ids = df_test['Id']



df_train.drop('Id', axis=1, inplace=True)

df_test.drop('Id', axis=1, inplace=True)
corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 12))

sns.heatmap(corrmat, vmax=.8, square=True)
sorted_sale_price_corrmat = corrmat['SalePrice'].sort_values()

print(sorted_sale_price_corrmat.head(20))

print("===================================")

print(sorted_sale_price_corrmat.tail(20))
k = 10

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

# np.corrcoefに行列をわたしたとき、行ごとの相関を調べる。

# 各cols同士の相関を知りたいので転置する

cm = np.corrcoef(df_train[cols].values.T)

plt.subplots(figsize=(9,9))

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, yticklabels=cols.values, xticklabels=cols.values)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', '1stFlrSF']

sns.pairplot(df_train[cols], size=2)
# 全データで調査する

all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

all_data.drop('SalePrice', axis=1, inplace=True)

print("all_data shape: {}".format(all_data.shape))
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum() / all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(35)
all_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1, inplace=True)

all_data.shape
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    all_data[col] = all_data[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
all_data['Utilities'].value_counts()
# ほぼ全部AllPub。意味がなさそうなので、列ごと消す

all_data.drop('Utilities', axis=1, inplace=True)
# 欠損しているものは、Typらしい。

all_data["Functional"] = all_data["Functional"].fillna("Typ")
cols = ['MSZoning','Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in cols:

    mode_val = all_data[col].mode()[0]

    all_data[col] = all_data[col].fillna(mode_val)
all_data.isnull().sum().sort_values(ascending=False)
sns.distplot(df_train['SalePrice'], fit=norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
df_train['SalePrice'] = np.log(df_train['SalePrice'])

sns.distplot(df_train['SalePrice'], fit=norm)

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
all_data['GrLivArea'] = np.log(all_data['GrLivArea'])

sns.distplot(all_data['GrLivArea'], fit=norm)

fig = plt.figure()

res = stats.probplot(all_data['GrLivArea'], plot=plt)
all_data['1stFlrSF'] = np.log(all_data['1stFlrSF'])

sns.distplot(all_data['1stFlrSF'], fit=norm)

fig = plt.figure()

res = stats.probplot(all_data['1stFlrSF'], plot=plt)
sns.distplot(all_data['TotalBsmtSF'], fit=norm)
# 0以外を対象にする

all_data.loc[all_data['TotalBsmtSF'] > 0, 'TotalBsmtSF'] = np.log(all_data['TotalBsmtSF'])

sns.distplot(all_data[all_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm)

fig = plt.figure()

res = stats.probplot(all_data[all_data['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
all_data.info()
# カテゴリ変数だが、intになっているものをstringに変換

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data = pd.get_dummies(all_data)
all_data.shape
all_data.columns.values
y = df_train['SalePrice']

train_data = all_data[:df_train.shape[0]]

test_data = all_data[df_test.shape[0]+1:]
train_data.shape
test_data.shape
from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import learning_curve

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data, y, random_state=42)
lasso = Lasso(alpha=0.0003)

lasso.fit(X_train, y_train)

lasso.score(X_test, y_test)
ridge = Ridge(alpha=3)

ridge.fit(X_train, y_train)

ridge.score(X_test, y_test)
lasso = Lasso(alpha=0.0003)

lasso.fit(train_data, y)

pred_y = lasso.predict(test_data)

pred_y = np.exp(pred_y)
submission = pd.DataFrame({

    'Id': test_ids,

    'SalePrice': pred_y

})

submission.to_csv('house_prices.csv', index=False)