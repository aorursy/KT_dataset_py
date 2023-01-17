import math

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import statsmodels.api as sm



from scipy.stats import pearsonr

from scipy.stats import mode



%matplotlib inline

sns.set(style='white', context='notebook', palette='deep')

plt.rcParams["figure.figsize"] = (15,7) # plot size
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
print('Train shape: ' + str(train.shape) + '.')

print('Test shape: ' + str(test.shape) + '.')
pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)
print('Missing values in the train set: ' + str(train.isnull().sum().sum()) + '.')

print('Missing values in the test set: ' + str(test.isnull().sum().sum()) + '.')
train['dataset'] = 'train'   # identify this as the train dataset

test['dataset'] = 'test'     # identify this as the train dataset

dataset = train.append(test, sort = False, ignore_index = True) # merge both datasets

del train, test              # free some memory.
dataset.shape
dataset.dataset.value_counts()
dataset.columns
stats = dataset.describe().T

for i in range(len(dataset.columns)):

    stats.loc[dataset.columns[i], 'mode'], stats.loc[dataset.columns[i], 'mode_count'] = mode(dataset[dataset.columns[i]])

    stats.loc[dataset.columns[i], 'unique_values'] = dataset[dataset.columns[i]].value_counts().size

    stats.loc[dataset.columns[i], 'NaN'] = dataset[dataset.columns[i]].isnull().sum()

    if np.isnan(stats.loc[dataset.columns[i], 'count']): 

        stats.loc[dataset.columns[i], 'count'] = dataset.shape[0] - stats.loc[dataset.columns[i], 'NaN']

stats = stats[['count', 'NaN', 'unique_values', 'mode', 'mode_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

stats.index.name = 'variable'

stats.reset_index(inplace = True)

stats
variables = list(stats[stats['NaN'] > 0].sort_values(by = ['NaN'], ascending = False).variable)

sns.barplot(x = 'variable', y='NaN', data = stats[stats['NaN'] > 0], order = variables)

plt.xticks(rotation=45)

stats[stats['NaN'] > 0].sort_values(by = ['NaN'], ascending = False)[['variable', 'NaN']]
dataset['MiscFeature'].fillna('NA', inplace = True)

dataset['Alley'].fillna('NA', inplace = True)

dataset['Fence'].fillna('NA', inplace = True)

dataset['FireplaceQu'].fillna('NA', inplace = True)

dataset['GarageFinish'].fillna('NA', inplace = True)

dataset['GarageQual'].fillna('NA', inplace = True)

dataset['GarageCond'].fillna('NA', inplace = True)

dataset['GarageType'].fillna('NA', inplace = True)

dataset['BsmtExposure'].fillna('NA', inplace = True)

dataset['BsmtCond'].fillna('NA', inplace = True)

dataset['BsmtQual'].fillna('NA', inplace = True)

dataset['BsmtFinType1'].fillna('NA', inplace = True)

dataset['BsmtFinType2'].fillna('NA', inplace = True)

dataset['BsmtFullBath'].fillna(0.0, inplace = True)

dataset['BsmtHalfBath'].fillna(0.0, inplace = True)

dataset['BsmtFinSF1'].fillna(0.0, inplace = True)

dataset['BsmtFinSF2'].fillna(0.0, inplace = True)

dataset['BsmtUnfSF'].fillna(0.0, inplace = True)

dataset['TotalBsmtSF'].fillna(0.0, inplace = True)
dataset.PoolQC.value_counts()
pd.crosstab(dataset.PoolArea, dataset.PoolQC)
dataset[(pd.isna(dataset['PoolQC'])) & (dataset['PoolArea'] > 0)].PoolArea.value_counts()
indexes = dataset[(pd.isna(dataset['PoolQC'])) & (dataset['PoolArea'] > 0)].index

dataset.loc[indexes, 'PoolQC'] = 'TA'

dataset['PoolQC'].fillna('NA', inplace = True)
#fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

ax1 = plt.subplot(212)

ax2 = plt.subplot(221)

ax3 = plt.subplot(222)

#plt.subplots_adjust(hspace = 0.5)



sns.scatterplot(y = 'LotFrontage', x = 'LotArea', data = dataset, ax = ax1, palette = 'rainbow')

sns.boxplot(y = 'LotFrontage', x = 'LotShape', data = dataset, ax = ax2, palette = 'rainbow')

sns.boxplot(y = 'LotFrontage', x = 'LotConfig', data = dataset, ax = ax3, palette = 'rainbow')
pearsonr(dataset.LotFrontage.dropna(), dataset[pd.notna(dataset['LotFrontage'])].LotArea)
pearsonr(dataset.LotFrontage.dropna(), np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/2))
pearsonr(dataset.LotFrontage.dropna(), np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4))
ax = sns.distplot(np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4))

ax.set(xlabel = 'Fourth root of LotArea')
ax = sns.regplot(y=dataset.LotFrontage.dropna(), x=np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4))

ax.set(xlabel = 'Fourth root of LotArea')
X = np.power(dataset[pd.notna(dataset['LotFrontage'])].LotArea, 1/4)

X = sm.add_constant(X)

model = sm.RLM(dataset.LotFrontage.dropna(), X)

results = model.fit()
index = dataset[pd.isna(dataset['LotFrontage'])].index

X_test = np.power(dataset.loc[index, 'LotArea'], 1/4)

X_test = sm.add_constant(X_test)

dataset.loc[index, 'LotFrontage'] = results.predict(X_test)
ax = sns.scatterplot(y=dataset.LotFrontage, x=np.power(dataset.LotArea, 1/4))

ax.set(xlabel = 'Fourth root of LotArea')
pearsonr(dataset.GarageYrBlt.dropna(), dataset[pd.notna(dataset['GarageYrBlt'])].YearBuilt)
sns.regplot(y = dataset.GarageYrBlt.dropna(), x = dataset[pd.notna(dataset['GarageYrBlt'])].YearBuilt)
index = dataset[dataset['GarageYrBlt'] > 2200].index

dataset.loc[index, 'GarageYrBlt'] = np.nan
# Fits the Regression Model.

X = dataset[pd.notna(dataset['GarageYrBlt'])]['YearBuilt']

X = sm.add_constant(X)

model = sm.OLS(dataset.GarageYrBlt.dropna(), X)

results = model.fit()
# Fill in the NaN values.

index = dataset[pd.isna(dataset['GarageYrBlt'])].index

X_test = dataset.loc[index, 'YearBuilt']

X_test = sm.add_constant(X_test)

X_test

dataset.loc[index, 'GarageYrBlt'] = round(results.predict(X_test),0).astype(int)
dataset[(dataset['GarageYrBlt'] < dataset['YearBuilt'])][['GarageYrBlt', 'YearBuilt']]
dataset['GarageYrBlt'] = np.where((dataset['GarageYrBlt'] >= 2000) & (dataset['GarageYrBlt'] == dataset['YearBuilt'] - 4), dataset['YearBuilt'], dataset['GarageYrBlt'])
dataset[(pd.notna(dataset['MasVnrArea'])) & (pd.isna(dataset['MasVnrType']))][['MasVnrArea', 'MasVnrType']]
dataset.groupby('MasVnrType', as_index = False)['MasVnrArea'].median()
index = dataset[(pd.notna(dataset['MasVnrArea'])) & (pd.isna(dataset['MasVnrType']))].index

dataset.loc[index, 'MasVnrType'] = 'Stone'
dataset['MasVnrType'].fillna('NA', inplace = True)

dataset['MasVnrArea'].fillna(0, inplace = True)
# LotArea and MSSubClass of the observations with NaN in the MSZoning variable.

dataset[pd.isna(dataset['MSZoning'])][['MSSubClass', 'LotArea']]
# median LotArea grouped by MSZoning and MSSubClass.

temp = dataset.groupby(['MSSubClass', 'MSZoning'], as_index=False)['LotArea'].median()

temp[temp['MSSubClass'].isin([20, 30, 70])]
# Makes the substitutions.

indexes = dataset[(pd.isna(dataset['MSZoning'])) & (dataset['MSSubClass'] == 30)].index

dataset.loc[indexes, 'MSZoning'] = 'C (all)'

indexes = dataset[pd.isna(dataset['MSZoning'])].index

dataset.loc[indexes, 'MSZoning'] = 'RL'
dataset['MSZoning'].value_counts()
dataset['Utilities'].value_counts()
dataset['Utilities'].fillna('AllPub', inplace = True)
dataset['Functional'].value_counts()
dataset['Functional'].fillna('Typ', inplace = True)
dataset['GarageArea'].value_counts()
dataset[pd.isna(dataset['GarageArea'])]
dataset[dataset['GarageType'] == 'Detchd'].GarageArea.describe()
dataset['GarageArea'].fillna(399, inplace = True)
dataset['GarageCars'].value_counts()
dataset[pd.isna(dataset['GarageCars'])]
temp = dataset.groupby(['GarageType', 'GarageCars'], as_index=False)['GarageArea'].median()

temp[temp['GarageType'] == 'Detchd']
dataset['GarageCars'].fillna(1, inplace = True)
dataset[pd.isna(dataset['Exterior2nd'])]
pd.crosstab(dataset['Exterior1st'], dataset['ExterCond'])
pd.crosstab(dataset['Exterior2nd'], dataset['ExterCond'])
len(dataset[dataset['Exterior1st'] == dataset['Exterior2nd']])
dataset['Exterior1st'].fillna('VinylSd', inplace = True)

dataset['Exterior2nd'].fillna('VinylSd', inplace = True)
dataset[pd.isna(dataset['KitchenQual'])]
dataset[dataset['KitchenAbvGr'] ==  1].KitchenQual.value_counts()
dataset['KitchenQual'].fillna('TA', inplace = True)
dataset['Electrical'].value_counts()
dataset['Electrical'].fillna('SBrkr', inplace = True)
dataset[pd.isna(dataset['SaleType'])]
dataset[dataset['SaleCondition'] == 'Normal'].SaleType.value_counts()
dataset['SaleType'].fillna('WD', inplace = True)
sns.distplot(dataset.SalePrice.dropna())
sns.distplot(np.log(dataset.SalePrice.dropna()), hist=True)
index = dataset[pd.notna(dataset['SalePrice'])].index

dataset.loc[index, 'SalePriceLog'] = np.log(dataset.loc[index, 'SalePrice'])
stats = dataset.describe().T

for i in range(len(dataset.columns)):

    stats.loc[dataset.columns[i], 'mode'], stats.loc[dataset.columns[i], 'mode_count'] = mode(dataset[dataset.columns[i]])

    stats.loc[dataset.columns[i], 'unique_values'] = dataset[dataset.columns[i]].value_counts().size

    stats.loc[dataset.columns[i], 'NaN'] = dataset[dataset.columns[i]].isnull().sum()

    if np.isnan(stats.loc[dataset.columns[i], 'count']): 

        stats.loc[dataset.columns[i], 'count'] = dataset.shape[0] - stats.loc[dataset.columns[i], 'NaN']

stats = stats[['count', 'NaN', 'unique_values', 'mode', 'mode_count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']]

stats.index.name = 'variable'

stats.reset_index(inplace = True)

stats
dataset['MoSold'] = dataset['MoSold'].astype(str)

dataset['MSSubClass'] = dataset['MSSubClass'].astype(str)

dataset['OverallCond'] = dataset['OverallCond'].astype(str)

dataset['OverallQual'] = dataset['OverallQual'].astype(str)
dataset['LotFrontageLog'] = np.log(dataset['LotFrontage'])

dataset['LotAreaLog'] = np.log(dataset['LotArea'])

dataset['1stFlrSFLog'] = np.log(dataset['1stFlrSF'])

dataset['GrLivAreaLog'] = np.log(dataset['GrLivArea'])
ax1 = plt.subplot(221)

ax2 = plt.subplot(222)

ax3 = plt.subplot(223)

ax4 = plt.subplot(224)



sns.distplot(dataset['LotFrontageLog'], ax = ax1)

sns.distplot(dataset['LotAreaLog'], ax = ax2)

sns.distplot(dataset['1stFlrSFLog'], ax = ax3)

sns.distplot(dataset['GrLivAreaLog'], ax = ax4)
dataset['2ndFlrDummy'] = np.where(dataset['2ndFlrSF'] > 0, 1, 0)

dataset['3SsnPorchDummy'] = np.where(dataset['3SsnPorch'] > 0, 1, 0)

dataset['AlleyDummy'] = np.where(dataset['Alley'] != 'NA', 1, 0)

dataset['EnclosedPorchDummy'] = np.where(dataset['EnclosedPorch'] > 0, 1, 0)

dataset['FireplaceDummy'] = np.where(dataset['FireplaceQu'] != 'NA', 1, 0)

dataset['LowQualFinDummy'] = np.where(dataset['LowQualFinSF'] > 0, 1, 0)

dataset['OpenPorchDummy'] = np.where(dataset['OpenPorchSF'] > 0, 1, 0)

dataset['PoolDummy'] = np.where(dataset['PoolQC'] != 'NA', 1, 0)

dataset['ScreenPorchDummy'] = np.where(dataset['ScreenPorch'] > 0, 1, 0)

dataset['PorchDummy'] = np.where(dataset['3SsnPorchDummy'] + dataset['EnclosedPorchDummy'] + dataset['OpenPorchDummy'] + dataset['ScreenPorchDummy'] > 0, 1, 0)

dataset['BsmtDummy'] = np.where(dataset['TotalBsmtSF'] > 0, 1, 0)

dataset['DeckDummy'] = np.where(dataset['WoodDeckSF'] > 0, 1, 0)
sns.heatmap(dataset.corr(), cmap="Blues", linewidths = .2)
dataset.corr()['SalePrice'].sort_values(ascending = False)
variables = list(dataset.columns)[1:80] + list(dataset.columns)[83:]



while len(variables) >= 8:

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3)

    plt.subplots_adjust(hspace = 0.5)

    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    for i in range(9):

        if type(dataset[variables[i]][0]) in [np.int64, np.float64]:

            sns.scatterplot(y = 'SalePriceLog', x = variables[i], data = dataset, ax = ax[i])

        else:

            sns.boxplot(y = 'SalePriceLog', x = variables[i], data = dataset, palette = 'rainbow', ax = ax[i])

    variables = variables[9:]    



fig, ((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3, figsize=(15, 4.5))

plt.subplots_adjust(hspace = 0.5)

sns.boxplot(y = 'SalePriceLog', x = variables[0], data = dataset, ax = ax1, palette = 'rainbow')

sns.boxplot(y = 'SalePriceLog', x = variables[1], data = dataset, ax = ax2, palette = 'rainbow')

sns.boxplot(y = 'SalePriceLog', x = variables[2], data = dataset, ax = ax3, palette = 'rainbow')

sns.boxplot(y = 'SalePriceLog', x = variables[3], data = dataset, ax = ax4, palette = 'rainbow')

sns.boxplot(y = 'SalePriceLog', x = variables[4], data = dataset, ax = ax5, palette = 'rainbow')
train = dataset[dataset['dataset'] == 'train'].copy()

train['dataset'] = None

test = dataset[dataset['dataset'] == 'test'].copy()

test['dataset'] = None
print('training set shape: ' + str(train.shape))

print('test set shape: ' + str(test.shape))
train.to_csv('train_mod.csv', index = False)

test.to_csv('test_mod.csv', index = False)