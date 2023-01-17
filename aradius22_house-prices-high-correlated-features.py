import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

import seaborn as sns

import os
test_data = pd.read_csv('../input/test.csv')

train_data = pd.read_csv('../input/train.csv')



# sizes of each doc

print(test_data.shape)

print(train_data.shape)
train_data.head(10)
df_train = pd.DataFrame(train_data)

df_test = pd.DataFrame(test_data)
df_train.describe(include='all')
cols_missed_train = df_train.isnull().sum()

cols_missed_valid = df_test.isnull().sum()



print('Columns with NaN in df_train: ', len(cols_missed_train[cols_missed_train > 0]))

print(cols_missed_train[cols_missed_train > 0].sort_values(ascending = False))



print('Columns with NaN in df_test: ', len(cols_missed_valid[cols_missed_valid > 0]))

print(cols_missed_valid[cols_missed_valid > 0].sort_values(ascending = False))
df_train['PoolQC'] = df_train['PoolQC'].fillna('None')

df_train['MiscFeature'] = df_train['MiscFeature'].fillna('None')

df_train['Alley'] = df_train['Alley'].fillna('None')

df_train['Fence'] = df_train['Fence'].fillna('None')

df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('None')

df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].mode()[0])

for col in ('GarageYrBlt', 'GarageCars', 'GarageArea'):

	df_train[col] = df_train[col].fillna(0)

for col in ('GarageType', 'GarageQual', 'GarageCond', 'GarageFinish'):

	df_train[col] = df_train[col].fillna('None')

for col in ('BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual'):

	df_train[col] = df_train[col].fillna('None')

df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mode()[0])

df_train['MasVnrType'] = df_train['MasVnrType'].fillna('None')

df_train['Electrical'] = df_train['Electrical'].fillna('None')

df_train['MSZoning'] = df_train['MSZoning'].fillna(df_train['MSZoning'].mode()[0])



df_train['Functional'] = df_train['Functional'].fillna(df_train['Functional'].mode()[0])

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_train[col] = df_train[col].fillna(0)

df_train['SaleType'] = df_train['SaleType'].fillna(df_train['SaleType'].mode()[0])

df_train['Utilities'] = df_train['Utilities'].fillna(df_train['Utilities'].mode()[0])

df_train['Exterior1st'] = df_train['Exterior1st'].fillna(df_train['Exterior1st'].mode()[0])

df_train['Exterior2nd'] = df_train['Exterior2nd'].fillna(df_train['Exterior2nd'].mode()[0])

df_train['KitchenQual'] = df_train['KitchenQual'].fillna(df_train['KitchenQual'].mode()[0])



# Checking nulls to be sure they are gone in df_train

print('Missed data in df_train: ', df_train.isnull().sum().sum())





# Replacing missing data in df_test

df_test['PoolQC'] = df_test['PoolQC'].fillna('None')

df_test['MiscFeature'] = df_test['MiscFeature'].fillna('None')

df_test['Alley'] = df_test['Alley'].fillna('None')

df_test['Fence'] = df_test['Fence'].fillna('None')

df_test['FireplaceQu'] = df_test['FireplaceQu'].fillna('None')

df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mode()[0])

for col in ('GarageYrBlt', 'GarageCars', 'GarageArea'):

	df_test[col] = df_test['GarageYrBlt'].fillna(0)

for col in ('GarageType', 'GarageQual', 'GarageCond', 'GarageFinish'):

	df_test[col] = df_test[col].fillna('None')

for col in ('BsmtFinType2', 'BsmtFinType1', 'BsmtExposure', 'BsmtCond', 'BsmtQual'):

	df_test[col] = df_test[col].fillna('None')

df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mode()[0])

df_test['MasVnrType'] = df_test['MasVnrType'].fillna('None')

df_test['Electrical'] = df_test['Electrical'].fillna('None')

df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])

df_test['Functional'] = df_test['Functional'].fillna(df_test['Functional'].mode()[0])

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    df_test[col] = df_test[col].fillna(0)

df_test['SaleType'] = df_test['SaleType'].fillna(df_test['SaleType'].mode()[0])

df_test['Utilities'] = df_test['Utilities'].fillna(df_test['Utilities'].mode()[0])

df_test['Exterior1st'] = df_test['Exterior1st'].fillna(df_test['Exterior1st'].mode()[0])

df_test['Exterior2nd'] = df_test['Exterior2nd'].fillna(df_test['Exterior2nd'].mode()[0])

df_test['KitchenQual'] = df_test['KitchenQual'].fillna(df_test['KitchenQual'].mode()[0])



# Checking nulls to be sure they are gone in df_train

print('Missed data in df_test: ', df_test.isnull().sum().sum())
df_train.dtypes.value_counts()
numerical_cols = df_train.select_dtypes(include = ['float64', 'int64'])

numerical_cols.head(10)
# Histogram for SalePrice

df_train.select_dtypes(include = ['float64', 'int64'])['SalePrice'].hist(bins=50, color='purple')
# Histogram with Density=1 to take analysis more accurate

numerical_cols['SalePrice'].hist(bins=50, density=1, color='DarkOrange')
# Let's take logarithm for more comfortable understandin our data

np.log(numerical_cols['SalePrice']).hist(bins=50, density=1, color='DarkCyan')

# As we can see we have outliers. We will get rid of them a bit later
for col in df_train.select_dtypes(include = ['object']):

    df_train[col] = df_train[col].astype('category')
# Histogram of SalePrice depending on MSZoning (normalized)

df_train.groupby('MSZoning')['SalePrice'].plot.hist(density=1, alpha=0.6)

plt.legend()
# Boxplot for analyzing more the one bunch of data

ax = df_train.boxplot(column='SalePrice', by='MSZoning')

# To make the title more readible

ax.get_figure().suptitle('')
numerical_cols.describe()
#Removing outliers (choosing data between first and third quartiles)

first_q = df_train['SalePrice'].describe()['25%']

third_q = df_train['SalePrice'].describe()['75%']

diff = third_q - first_q



cols_train = df_train[(df_train['SalePrice'] > (first_q - 3 * diff))&

                     (df_train['SalePrice'] < (third_q + 3 * diff))]

print('Removed outliers: ' + str(len(df_train) - len(cols_train)) + ' objects')
# Building correlation matrix for understanding how the characteristics influence to each other

n_df = cols_train.copy()

corr_matrix = n_df.drop('Id', axis=1).corr()

f, ax = plt.subplots(figsize=(16, 12))

plt.title('Correlation Matrix',fontsize=22)

sns.heatmap(corr_matrix, vmax=1, center=0, annot=True, fmt='.1f')
pos_cor = corr_matrix.SalePrice[corr_matrix.SalePrice.values > 0.40].sort_values(ascending=False)[1:]

neg_cor = corr_matrix.SalePrice[corr_matrix.SalePrice.values < 0.0].sort_values(ascending=True)[1:]

features = pos_cor.append(neg_cor).index

features
test_id = df_test['Id']
# Removing data with a high correlation level

cols_train.drop(columns=['Id'], axis=1, inplace=True)

df_test.drop(columns=['Id'], axis=1, inplace=True)
# Aggregation of number of houses by each of neighborhoods and mean prices per each neighborhood (for understanding of mean prices in common)

cols_train.groupby('Neighborhood').agg({'Neighborhood':'size','SalePrice':'mean'}).rename(columns={'Neighborhood':'hous count','SalePrice':'mean price'})
cols_train['SalePrice'] = np.log(cols_train['SalePrice'])
SalePrice = cols_train['SalePrice']
new_train = cols_train[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt',

       'FullBath', 'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'MasVnrArea',

       'EnclosedPorch', 'MSSubClass', 'OverallCond', 'LowQualFinSF', 'YrSold',

       'BsmtHalfBath', 'MiscVal', 'BsmtFinSF2']]

new_test = df_test[['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt',

       'FullBath', 'YearRemodAdd', 'TotRmsAbvGrd', 'Fireplaces', 'MasVnrArea',

       'EnclosedPorch', 'MSSubClass', 'OverallCond', 'LowQualFinSF', 'YrSold',

       'BsmtHalfBath', 'MiscVal', 'BsmtFinSF2']]
print(new_train.shape)

print(new_test.shape)
y = SalePrice

X = new_train



# Split dataset to train and valid for training and testing

train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)
train_model = RandomForestRegressor()

train_model.fit(train_X, train_y)



val_pred = train_model.predict(valid_X)



rmse = np.sqrt(mean_squared_error(valid_y, val_pred))

rmse
final_pred = train_model.predict(new_test.values)

final_pred = np.exp(final_pred)
df_pred = pd.DataFrame({"id":test_id, "SalePrice":final_pred})

df_pred.SalePrice = df_pred.SalePrice.round(0)

df_pred.to_csv('submission.csv', sep=',', encoding='utf-8', index=False)
