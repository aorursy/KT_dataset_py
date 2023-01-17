import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from scipy import stats

from scipy.stats import norm, skew



df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', encoding='utf-8')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv',encoding='utf-8')



df_train.head()

df_test.head()
df_train['SalePrice'].describe()
sns.distplot(df_train['SalePrice'], color='blue')
print("Skew is : %f" % df_train['SalePrice'].skew())

print("Kurtosis is: %f" % df_train['SalePrice'].kurt())
sns.distplot(df_train['SalePrice'], fit=stats.norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
corrmat = df_train.corr()

k = 10 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
nulls = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([nulls, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
df_test = df_test.drop(['PoolQC'],1)

df_test = df_test.drop(['MiscFeature'],1)

df_test = df_test.drop(['Alley'],1)

df_test = df_test.drop(['Fence'],1)

df_test = df_test.drop(['FireplaceQu'],1)



df_train = df_train.drop(['PoolQC'],1)

df_train = df_train.drop(['MiscFeature'],1)

df_train = df_train.drop(['Alley'],1)

df_train = df_train.drop(['Fence'],1)

df_train = df_train.drop(['FireplaceQu'],1)



nulls = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([nulls, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    df_train[col] = df_train[col].fillna('None')

    df_test[col] = df_test[col].fillna('None')



for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    df_train[col] = df_train[col].fillna(0)

    df_test[col] = df_test[col].fillna(0)



for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_train[col] = df_train[col].fillna(0)

    df_test[col] = df_test[col].fillna(0)



for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    df_train[col] = df_train[col].fillna('None')

    df_test[col] = df_test[col].fillna('None')



df_test["MasVnrType"] = df_test["MasVnrType"].fillna("None")

df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")

df_test["MasVnrArea"] = df_test["MasVnrArea"].fillna(0)

df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)



df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])



df_test["Functional"] = df_test["Functional"].fillna("Typ")



df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mode()[0])

df_train['LotFrontage'] =df_train['LotFrontage'].fillna(df_train['LotFrontage'].mode()[0])



df_test['Electrical'] = df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])

df_train['Electrical'] =df_train['Electrical'].fillna(df_train['Electrical'].mode()[0])



df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])



df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])

df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])



df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])



nulls = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([nulls, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(15)
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,3), sharey=True, dpi=120)

fig, (ax3, ax4) = plt.subplots(1,2, figsize=(12,3), sharey=True, dpi=120)





ax1.plot(df_train['TotalBsmtSF'], df_train['SalePrice'], 'bo')

ax2.plot(df_train['GrLivArea'], df_train['SalePrice'], 'bo')

ax3.plot(df_train['OverallQual'], df_train['SalePrice'], 'bo')

ax4.plot(df_train['GarageCars'], df_train['SalePrice'], 'bo')



ax1.set_ylabel('SalePrice')

ax3.set_ylabel('SalePrice')

ax1.set_xlabel('TotalBsmtSF')

ax2.set_xlabel('GrLivArea')

ax3.set_xlabel('OverallQual')

ax4.set_xlabel('GarageCars')
df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)

df_train = df_train.drop(df_train[df_train['Id'] == 30].index)

df_train = df_train.drop(df_train[df_train['Id'] == 88].index)

df_train = df_train.drop(df_train[df_train['Id'] == 462].index)

df_train = df_train.drop(df_train[df_train['Id'] == 631].index)

df_train = df_train.drop(df_train[df_train['Id'] == 1322].index)

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,3), sharey=True, dpi=120)

fig, (ax3, ax4) = plt.subplots(1,2, figsize=(12,3), sharey=True, dpi=120)



ax1.plot(df_train['TotalBsmtSF'], df_train['SalePrice'], 'bo')

ax1.set_xlim(left = 0,right = 6000)

ax2.plot(df_train['GrLivArea'], df_train['SalePrice'], 'bo')

ax2.set_xlim(left = 0,right = 5000)

ax3.plot(df_train['OverallQual'], df_train['SalePrice'], 'bo')

ax4.plot(df_train['GarageCars'], df_train['SalePrice'], 'bo')



ax1.set_ylabel('SalePrice')

ax3.set_ylabel('SalePrice')

ax1.set_xlabel('TotalBsmtSF')

ax2.set_xlabel('GrLivArea')

ax3.set_xlabel('OverallQual')

ax4.set_xlabel('GarageCars')
sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])

df_test['GrLivArea'] = np.log(df_test['GrLivArea'])



sns.distplot(df_train['GrLivArea'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['GrLivArea'], plot=plt)
y_train = df_train.SalePrice.values

df_train.drop(['SalePrice'], axis=1, inplace=True)

df_train.drop("Id", axis = 1, inplace = True)

df_test.drop("Id", axis = 1, inplace = True)



m = len(df_train)
all_data = pd.concat((df_train,df_test),sort=True)



quantitative = [f for f in all_data.columns if all_data.dtypes[f] != 'object']

categoricals = df_train.select_dtypes(exclude=[np.number])

print('We have :',len(categoricals.columns),' categoricals variables')

print('And :',len(quantitative), 'quantitative variables')
all_data = pd.get_dummies(all_data)



quantitative = [f for f in all_data.columns if all_data.dtypes[f] != 'object']

categoricals = all_data.select_dtypes(exclude=[np.number])

print('We have :',len(categoricals.columns),' categoricals variables')

print('And :',len(quantitative), 'quantitative variables')



df_train = all_data.iloc[0:m, :]

df_test = all_data.iloc[m:, :]