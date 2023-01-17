import pandas as pd

import matplotlib.pyplot as plt

import numpy as np
train_df = pd.read_csv('../input/train.csv', index_col=0)

test_df = pd.read_csv('../input/test.csv', index_col=0)

train_df.head()
# 查看当前特征的类型

train_df['MSSubClass'] = train_df['MSSubClass'].astype(str)

test_df['MSSubClass'] = test_df['MSSubClass'].astype(str)

numeric_col_names = train_df.columns[train_df.dtypes != 'object']

object_col_names = train_df.columns[train_df.dtypes == 'object']

print('numeric col : ', numeric_col_names, len(numeric_col_names))

print('object col : ' , object_col_names, len(object_col_names))

# 潜在的分类型的特征变量， overallqual, overallcond, moSold, YrSold
# 使用平均值替换train_df 中的 nan,删除

print(train_df[numeric_col_names].isna().sum().sort_values(ascending=False).head(5))

cols_mean = train_df.loc[:, numeric_col_names].mean()

train_df.loc[:, numeric_col_names] = pd.DataFrame(

    train_df.loc[:, numeric_col_names].fillna(cols_mean))

print(train_df[numeric_col_names].isna().sum().sort_values(ascending=False).head(5))
plt.figure(figsize=(12, 12))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.title(numeric_col_names[i])

    plt.scatter(train_df[numeric_col_names[i]], train_df['SalePrice'])

plt.show()
plt.figure(figsize=(12, 12))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.title(numeric_col_names[i+9])

    plt.scatter(train_df[numeric_col_names[i+9]], train_df['SalePrice'])

plt.show()
plt.figure(figsize=(12, 12))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.title(numeric_col_names[i+18])

    plt.scatter(train_df[numeric_col_names[i+18]], train_df['SalePrice'])

plt.show()
plt.figure(figsize=(12, 12))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.title(numeric_col_names[i+27])

    plt.scatter(train_df[numeric_col_names[i+27]], train_df['SalePrice'])

plt.show()
import seaborn as sns

plt.figure(figsize=(14, 30))

corr1 = train_df.corr()

plt.subplot(3,1,1)

plt.title('persion')

sns.heatmap(corr1, vmax=.8, square=True)

corr2 = train_df.corr('kendall')

plt.subplot(3,1,2)

plt.title('kendall')

sns.heatmap(corr2, vmax=.8, square=True)

corr3 = train_df.corr('spearman')

plt.subplot(3,1,3)

plt.title('spearman')

sns.heatmap(corr3, vmax=.8, square=True)

plt.show()
print('objetc col names : ', object_col_names, len(object_col_names))
print(train_df[object_col_names].isna().sum().sort_values(ascending=False).head(5))

train_df.loc[:, object_col_names] = train_df.loc[:, object_col_names].fillna('No')

print(train_df[object_col_names].isna().sum().sort_values(ascending=False).head(5))
plt.figure(figsize=(12,12))

for i in range(16):

    col_name = object_col_names[i]

    type_names = train_df[col_name].value_counts().keys()

    color_nums = np.linspace(0, 1, len(type_names))

    color_map = {t: c for t,c in zip(type_names, color_nums)}

    color_seq = [color_map[t] for t in train_df[col_name]]

    plt.subplot(4, 4, i+1)

    plt.title(col_name)

    plt.scatter(np.log1p(train_df['SalePrice']), color_seq, c=color_seq)
plt.figure(figsize=(12,12))

for i in range(16):

    col_name = object_col_names[i+16]

    type_names = train_df[col_name].value_counts().keys()

    color_nums = np.linspace(0, 1, len(type_names))

    color_map = {t: c for t,c in zip(type_names, color_nums)}

    color_seq = [color_map[t] for t in train_df[col_name]]

    plt.subplot(4, 4, i+1)

    plt.title(col_name)

    plt.scatter(np.log1p(train_df['SalePrice']), color_seq, c=color_seq)
plt.figure(figsize=(12,12))

for i in range(12):

    col_name = object_col_names[i+32]

    type_names = train_df[col_name].value_counts().keys()

    color_nums = np.linspace(0, 1, len(type_names))

    color_map = {t: c for t,c in zip(type_names, color_nums)}

    color_seq = [color_map[t] for t in train_df[col_name]]

    plt.subplot(4, 4, i+1)

    plt.title(col_name)

    plt.scatter(np.log1p(train_df['SalePrice']), color_seq, c=color_seq)
# 转化类别

col_should_be_type = ['OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 

                      'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'MoSold', 'YrSold']

for col in col_should_be_type:

    train_df[col] = train_df[col].astype(str)
# 删除分布不好的变量

col_should_be_del = ['BsmtFinSF2', 'LowQualFinSF', '3SsnPorch', 

                     'PoolArea', 'MiscVal', 'PoolQC', 'Alley', 'Utilities', 'Street']

train_df.drop(col_should_be_del, axis=1, inplace=True)

train_df.head()
# 去除奇异值  

"""

LotFrange > 200, LotArea > 100000, BsmtFinSF1 > 3000, 

BsmtUnfSF, TotalBsmtSF, GrLivArea, TotRmsAbvGrd, OpenPorchSF

"""



train_df.loc[train_df['LotFrontage'] > 200, 'LotFrontage'] = 200

train_df.loc[train_df['LotArea'] > 100000, 'LotArea'] = 100000

train_df.loc[train_df['BsmtFinSF1'] > 3000, 'BsmtFinSF1'] = 100000

# train_df.loc[train_df['BsmtUnfSF'] < 1500 & train_df['SalePrice'] > 600000, 'SalePrice'] = 600000

train_df.loc[train_df['TotalBsmtSF'] > 3000, 'TotalBsmtSF'] = 3000

train_df.loc[train_df['GrLivArea'] > 4000, 'GrLivArea'] = 4000

train_df.loc[train_df['TotRmsAbvGrd'] > 12, 'TotRmsAbvGrd'] = 12

train_df.loc[train_df['OpenPorchSF'] > 400, 'OpenPorchSF'] = 400
train_price = np.log1p(train_df.pop('SalePrice'))
numeric_col_names = train_df.columns[train_df.dtypes != 'object']

object_col_names = train_df.columns[train_df.dtypes == 'object']

print('numeric col : ', numeric_col_names, len(numeric_col_names))

print('object col : ' , object_col_names, len(object_col_names))
# 数值类型的变量标准化

cols_numeric_mean = train_df.loc[:, numeric_col_names].mean()

cols_numeric_std = train_df.loc[:, numeric_col_names].std()

train_df.loc[:, numeric_col_names] = (train_df.loc[:, numeric_col_names] - cols_numeric_mean) / cols_numeric_std
# one-hot

print(train_df.shape)

train_df = pd.get_dummies(train_df)

print(train_df.shape)

train_df.head()
# from xgboost import XGBRegressor

# xgboost = XGBRegressor(max_depth=5)



train_data_x = train_df.loc[:1000, :]

train_data_y = train_price.loc[:1000]

test_data_x = train_df.loc[1000:, :]

test_data_y = train_price.loc[1000:]

plt.subplot(2,1,1)

train_data_y.hist()

plt.subplot(2,1,2)

test_data_y.hist()

plt.show()

print('train shape: ',train_data_x.shape, train_data_y.shape)

print('test shape: ', test_data_x.shape, test_data_y.shape)

print(train_data_y.isna().sum())

print(test_data_y.isna().sum())
from xgboost import XGBRegressor



xgboost = XGBRegressor(max_depth=5)

xgboost.fit(train_data_x, train_data_y)



prediction = xgboost.predict(test_data_x)

score = np.sqrt(np.mean(np.square(prediction - test_data_y)))

print(score)
from sklearn.linear_model import Ridge

from sklearn.ensemble import BaggingRegressor



ridge = Ridge(18)

bagging = BaggingRegressor(n_estimators=18, base_estimator=ridge)

bagging.fit(train_data_x, train_data_y)

prediction = bagging.predict(test_data_x)

score = np.sqrt(np.mean(np.square(prediction - test_data_y)))

print(score)