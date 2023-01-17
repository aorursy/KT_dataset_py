import numpy as np

import pandas as pd

import sklearn

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from scipy import stats
data_df = pd.read_csv('../input/train.csv')

data_df.head()
test_data_df = pd.read_csv('../input/test.csv')
data_df.shape
data_df.columns
train_df = data_df.copy()

test_df = test_data_df.copy()
(data_df['SalePrice'].min(), data_df['SalePrice'].max())
data_df['SalePrice'].mean()
data_df['SalePrice'].median()
data_df['SalePrice'].plot.hist()
np.log1p(data_df['SalePrice']).plot.hist()
plt.subplots(figsize=(20,15))

sns.heatmap(data=data_df.corr())
data_df['LotFrontage'].isnull().sum()
sns.regplot(data_df['LotArea'], data_df['SalePrice'])
data_df['OverallQual'].plot.hist()
sns.regplot(data_df['OverallQual'], data_df['SalePrice'])
data_df['YearBuilt'].plot.hist()
(data_df['YearBuilt'].min(), data_df['YearBuilt'].max())
sns.regplot(data_df['YearBuilt'], data_df['SalePrice'])
sns.regplot(data_df['TotalBsmtSF'], data_df['SalePrice'])
data_df[data_df['TotalBsmtSF'] > 6000]
outliers = set(data_df[data_df['TotalBsmtSF'] > 6000].index.values)



print('Running list of outliers: ', outliers)
sns.regplot(data_df['1stFlrSF'], data_df['SalePrice'])
data_df[data_df['1stFlrSF'] > 4000]
sns.regplot(data_df['GrLivArea'], data_df['SalePrice'])
data_df[data_df['GrLivArea'] > 4000].loc[:, ['GrLivArea', 'SalePrice']]
zscore = stats.zscore(data_df['GrLivArea'])

thresh = 4

print(np.where(zscore > thresh), zscore[np.where(zscore > thresh)])
outliers.update([outlier for outlier in list(np.where(data_df['GrLivArea'] > 4000)[0]) if outlier not in outliers])

print('Running list of outliers: ', outliers)
missing_data = (data_df.isnull().sum()/len(data_df)*100).sort_values(ascending=False)

plt.figure(figsize=(25, 12))

plt.xticks(rotation="90")

sns.barplot(missing_data.index, missing_data)
data_df[data_df['PoolArea'] != 0].loc[:, ['PoolArea', 'PoolQC']]
train_df['PoolQC'].fillna('None', inplace=True)
test_df['PoolQC'].isnull().sum()
test_df[test_df['PoolArea'] != 0].loc[:, ['PoolArea', 'PoolQC']]
test_df[(test_df['PoolArea'] != 0) & (test_df['PoolQC'].isnull())].loc[:, ['PoolQC']].fillna('Fa', inplace=True)

test_df['PoolQC'].fillna('None', inplace=True)
data_df['MiscVal'].plot.hist()
train_df.drop(columns=['MiscFeature'], inplace=True)
print('Number of missing values in MiscFeature in test data = ', test_df['MiscFeature'].isnull().sum())



test_df['MiscVal'].plot.hist()
test_df.drop(columns=['MiscFeature'], inplace=True)
train_df['Alley'].fillna('None', inplace=True)

test_df['Alley'].fillna('None', inplace=True)
train_df.drop(columns=['Fence'], inplace=True)

test_df.drop(columns=['Fence'], inplace=True)
print('Number of missing values in FireplaceQu = ', data_df['FireplaceQu'].isnull().sum())

print('Number of missing values in Fireplaces = ', data_df['Fireplaces'].isnull().sum())
data_df['Fireplaces'].plot.hist()
len(data_df[(data_df['Fireplaces']!=0) & (data_df['FireplaceQu'].isnull())])
len(test_df[(test_df['Fireplaces']!=0) & (test_df['FireplaceQu'].isnull())])
train_df['FireplaceQu'].fillna('None', inplace=True)
test_df['FireplaceQu'].fillna('None', inplace=True)
temp = data_df[data_df['LotArea'] < 55000]

sns.regplot(temp['LotArea'], temp['SalePrice'])
data_df['LotFrontage'].mean()
data_df['LotFrontage'].median()
data_df['LotFrontage'].plot.hist()
train_df['LotFrontage'].fillna(data_df['LotFrontage'].median(), inplace=True)
print('Number of missing values for LotFrontage in test set = ', test_df['LotFrontage'].isnull().sum())

print('Mean = ', test_df['LotFrontage'].mean())

print('Median = ', test_df['LotFrontage'].median())
test_df['LotFrontage'].plot.hist()
test_df['LotFrontage'].fillna(test_df['LotFrontage'].median(), inplace=True)
sns.regplot(data_df['LotFrontage'], data_df['SalePrice'])
sns.boxplot(data_df['LotFrontage'])
zscore = np.abs(stats.zscore(data_df['LotFrontage']))



thresh = 4

print(np.where(zscore > thresh), zscore[np.where(zscore > thresh)])
data_df[data_df['LotFrontage'] > 300]
outliers.update([outlier for outlier in list(np.where(data_df['LotFrontage'] > 300)[0]) if outlier not in outliers])



print('Running list of outliers: ', outliers)
sns.boxplot(test_df['LotFrontage'])
data_df['GarageArea'].isnull().sum(), data_df['GarageArea'].mean(), data_df['GarageArea'].median()
data_df['GarageArea'].plot.hist()
# Check if there is any house with missing values in the garage category with nonzero garage area

len(data_df[(data_df['GarageArea']!=0) & ((data_df['GarageCond'].isnull()) | 

                                          (data_df['GarageType'].isnull()) | 

                                          (data_df['GarageYrBlt'].isnull()) | 

                                          (data_df['GarageFinish'].isnull()) | 

                                          (data_df['GarageQual'].isnull()))])
train_df['GarageCond'].fillna('None', inplace=True)

train_df['GarageType'].fillna('None', inplace=True)

train_df['GarageYrBlt'].fillna('None', inplace=True)

train_df['GarageFinish'].fillna('None', inplace=True)

train_df['GarageQual'].fillna('None', inplace=True)
print('Number of missing values in garage area = ', test_df['GarageArea'].isnull().sum())
test_df[(test_df['GarageArea']!=0) & ((test_df['GarageType'].isnull()) | 

                                      (test_df['GarageYrBlt'].isnull()) | 

                                      (test_df['GarageFinish'].isnull()) | 

                                      (test_df['GarageQual'].isnull())

                                     )].loc[:, ['GarageArea', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual']]
test_df['GarageArea'].fillna(test_df['GarageArea'].median(), inplace=True)
print('Year most garages were built: ', test_df['GarageYrBlt'].median())

print('Year house 666 was built and year it was remodelled: ', test_df.at[666, 'YearBuilt'], test_df.at[666, 'YearRemodAdd'])

print('Year house 666 was built and year it was remodelled: ', test_df.at[1116, 'YearBuilt'], test_df.at[1116, 'YearRemodAdd'])
test_df['GarageFinish'].value_counts()
test_df['GarageQual'].value_counts()
test_df.at[666, 'GarageYrBlt'] = 1983

test_df.at[1116, 'GarageYrBlt'] = 1999



test_df.at[666, 'GarageFinish'] = 'Unf'

test_df.at[1116, 'GarageFinish'] = 'Unf'



test_df.at[666, 'GarageQual'] = 'TA'

test_df.at[1116, 'GarageQual'] = 'TA'
test_df['GarageCond'].fillna('None', inplace=True)

test_df['GarageType'].fillna('None', inplace=True)

test_df['GarageYrBlt'].fillna('None', inplace=True)

test_df['GarageFinish'].fillna('None', inplace=True)

test_df['GarageQual'].fillna('None', inplace=True)
train_df.isnull().sum().sum()
(data_df['BsmtExposure'].isnull().sum(),

data_df['BsmtFinType2'].isnull().sum(),

data_df['BsmtFinType1'].isnull().sum(),

data_df['BsmtCond'].isnull().sum(),

data_df['BsmtQual'].isnull().sum())
data_df['TotalBsmtSF'].isnull().sum()
data_df[(data_df['TotalBsmtSF'] != 0) & ((data_df['BsmtExposure'].isnull()) | 

                                         (data_df['BsmtFinType2'].isnull()) | 

                                         (data_df['BsmtFinType1'].isnull()) |

                                         (data_df['BsmtCond'].isnull()) |

                                         (data_df['BsmtQual'].isnull())

                                        )].loc[:, ['TotalBsmtSF', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']]
data_df['BsmtExposure'].value_counts()
data_df['BsmtFinType2'].value_counts()
train_df.at[332, 'BsmtFinType2'] = 'Unf'

train_df.at[948, 'BsmtExposure'] = 'No'
train_df['BsmtExposure'].fillna('None', inplace=True)

train_df['BsmtFinType2'].fillna('None', inplace=True)

train_df['BsmtFinType1'].fillna('None', inplace=True)

train_df['BsmtCond'].fillna('None', inplace=True)

train_df['BsmtQual'].fillna('None', inplace=True)
(test_df['BsmtExposure'].isnull().sum(),

test_df['BsmtFinType2'].isnull().sum(),

test_df['BsmtFinType1'].isnull().sum(),

test_df['BsmtCond'].isnull().sum(),

test_df['BsmtQual'].isnull().sum(),

test_df['TotalBsmtSF'].isnull().sum())
test_df[(test_df['TotalBsmtSF'] != 0) & ((test_df['BsmtExposure'].isnull()) | 

                                         (test_df['BsmtFinType2'].isnull()) | 

                                         (test_df['BsmtFinType1'].isnull()) |

                                         (test_df['BsmtCond'].isnull()) |

                                         (test_df['BsmtQual'].isnull())

                                        )].loc[:, ['TotalBsmtSF', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual']]
basement = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

            'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

test_df.loc[660, basement]
test_df.at[660, 'BsmtQual'] = None

test_df.at[660, 'BsmtCond'] = None

test_df.at[660, 'BsmtExposure'] = None

test_df.at[660, 'BsmtFinType1'] = None

test_df.at[660, 'BsmtFinSF1'] = 0

test_df.at[660, 'BsmtFinType2'] = None

test_df.at[660, 'BsmtFinSF2'] = 0

test_df.at[660, 'BsmtUnfSF'] = 0

test_df.at[660, 'TotalBsmtSF'] = 0

test_df.at[660, 'BsmtFullBath'] = None

test_df.at[660, 'BsmtHalfBath'] = None
train_df['BsmtCond'].value_counts()
train_df['BsmtQual'].value_counts()
test_df.at[27, 'BsmtExposure'] = 'No'

test_df.at[580, 'BsmtCond'] = 'TA'

test_df.at[725, 'BsmtCond'] = 'TA'

test_df.at[757, 'BsmtQual'] = 'TA'

test_df.at[758, 'BsmtQual'] = 'TA'

test_df.at[888, 'BsmtExposure'] = 'No'

test_df.at[1064, 'BsmtCond'] = 'TA'
test_df['BsmtExposure'].fillna('None', inplace=True)

test_df['BsmtFinType2'].fillna('None', inplace=True)

test_df['BsmtFinType1'].fillna('None', inplace=True)

test_df['BsmtCond'].fillna('None', inplace=True)

test_df['BsmtQual'].fillna('None', inplace=True)
(data_df['MasVnrType'].isnull().sum(), data_df['MasVnrArea'].isnull().sum())
data_df[(data_df['MasVnrType'].isnull()) | (data_df['MasVnrArea'].isnull())].loc[:, ['MasVnrType', 'MasVnrArea']]
data_df['MasVnrArea'].mean()
data_df['MasVnrArea'].median()
data_df['MasVnrArea'].plot.hist()
len(data_df[data_df['MasVnrArea'] == 0])/len(data_df['MasVnrArea'])
train_df['MasVnrArea'].fillna(0, inplace=True)
train_df['MasVnrType'].fillna('None', inplace=True)
test_df['MasVnrArea'].plot.hist()
print('Number of missing values in MasVnrType ', test_df['MasVnrType'].isnull().sum(), 'and MasVnArea ', test_df['MasVnrArea'].isnull().sum())
test_df[(test_df['MasVnrType'].isnull()) |

        (test_df['MasVnrArea'].isnull())].loc[:, ['MasVnrType', 'MasVnrArea']]
test_df['MasVnrType'].value_counts()
test_df.at[1150, 'MasVnrType'] = 'BrkFace'

test_df['MasVnrType'].fillna('None', inplace=True)

test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].median(), inplace=True)
print('Number of missing values = ', data_df['Electrical'].isnull().sum())
data_df['Electrical'].value_counts()
train_df['Electrical'].fillna('Sbrkr', inplace=True)
print('Number of missing values = ', test_df['Electrical'].isnull().sum())
train_df.isnull().sum().sum()
test_df.columns[np.where(test_df.isnull().sum() != 0)]
test_df['MSZoning'].fillna('RL', inplace=True)

test_df['Utilities'].fillna('AllPub', inplace=True)

test_df['Exterior1st'].fillna('VinylSd', inplace=True)

test_df['Exterior2nd'].fillna('VinylSd', inplace=True)

test_df['BsmtFullBath'].fillna(0, inplace=True)

test_df['BsmtHalfBath'].fillna(0, inplace=True)

test_df['KitchenQual'].fillna('TA', inplace=True)

test_df['Functional'].fillna('Typ', inplace=True)

test_df['GarageCars'].fillna(2, inplace=True)

test_df['SaleType'].fillna('WD', inplace=True)
test_df.isnull().sum().sum()
train_df.drop(columns=['Id', 'SalePrice'], inplace=True)

test_df.drop(columns=['Id'], inplace=True)
train_df.shape, test_df.shape
print('Current list of outliers ', list(outliers))
train_df.drop(index=list(outliers), inplace=True)
data_df.drop(index=list(outliers), inplace=True)
dummies = pd.get_dummies(pd.concat((train_df, test_df), axis=0))
dummies.shape
X = dummies.iloc[:train_df.shape[0]]

X_test = dummies.iloc[train_df.shape[0]:]
X.shape, X_test.shape
y = np.log(data_df['SalePrice'] + 1)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score



GBR3 = GradientBoostingRegressor(learning_rate= 0.1, n_estimators=150, max_depth=4)

cv_score = cross_val_score(GBR3, X, y, cv=5, scoring='neg_mean_squared_error')

print('Cross validation scores for GBR model:', np.sqrt(-cv_score).mean())
from sklearn.ensemble import RandomForestRegressor

RF3 = RandomForestRegressor(n_estimators=100, max_features=20, random_state=0)

cv_score2 = cross_val_score(RF3, X, y, cv=5, scoring='neg_mean_squared_error')

print('Cross validation scores for GBR model:', np.sqrt(-cv_score2).mean())
GBR3.fit(X, y)

y_pred = np.exp(GBR3.predict(X_test)) - 1

answer = pd.DataFrame(data=y_pred, columns=['SalePrice'])

answer.insert(loc=0, column='Id', value=test_data_df['Id'])



answer.to_csv('submission.csv', index=False)