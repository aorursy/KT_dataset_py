# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



import scipy.stats as stats

from scipy.special import boxcox1p



import statsmodels.api as sm



from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import (LinearRegression, Lasso, Ridge, ElasticNet,

                                  LassoCV, RidgeCV, ElasticNetCV)

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.feature_selection import RFECV



from xgboost import XGBRegressor



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
folder = '/kaggle/input/house-prices-advanced-regression-techniques/'

train = pd.read_csv(folder+'train.csv')

test = pd.read_csv(folder+'test.csv')

print(train.shape)

print(test.shape)
train_ids = train['Id'].tolist()

test_ids = test['Id'].tolist()

data = pd.concat([train, test])

data = data.set_index('Id')

data.shape
fig, ax = plt.subplots(ncols=2, figsize=(16, 6))

sns.distplot(data['SalePrice'], ax=ax[0]).set_title('Skew: {}'.format(round(data['SalePrice'].skew(),3)))

sns.distplot(np.log(data['SalePrice']), ax=ax[1]).set_title('Skew: {}'.format(round(np.log(data['SalePrice']).skew(),3)));
data['SalePrice_log'] = np.log(data['SalePrice'])

data = data.drop(columns=['SalePrice'])
corr_price = abs(data.corr()['SalePrice_log']).sort_values(ascending=False)

high_corr_price = corr_price[corr_price > 0.5].index

plt.figure(figsize=(15,10))

sns.heatmap(data[high_corr_price].corr(), annot=True)
plt.figure(figsize=(15, 6))

sns.boxplot(x='OverallQual', y='SalePrice_log', data=data)
sns.scatterplot(x='GrLivArea', y='SalePrice_log', data=data)
cat_vars = []

ord_vars = []

num_vars = []

qualities_map = {'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}



def get_map_to_ordinal(ordered_categories):

    return dict(zip(ordered_categories, range(len(ordered_categories))))
data.loc[:,data.isna().any()].isna().sum()
miss_many_vars = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']

miss_5_idx = data.index[data[miss_many_vars].isna().all(axis=1)]

zero_sf_idx = data.index[(data['TotalBsmtSF'] == 0) | (data['TotalBsmtSF'].isna())]

set(miss_5_idx) == set(zero_sf_idx)
miss_few_vars = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']

data[miss_few_vars].dtypes
data[miss_few_vars] = data[miss_few_vars].fillna(0)

num_vars.extend(miss_few_vars)
for var in miss_many_vars:

    print(var, data[var].unique())
for var in miss_many_vars:

    data.loc[miss_5_idx, var] = data.loc[miss_5_idx, var].fillna('NA')
data[miss_many_vars] = data[miss_many_vars].fillna(data.mode().iloc[0])

for var in miss_many_vars:

    print(var, data[var].unique())
data = data.replace({'BsmtQual': qualities_map, 'BsmtCond': qualities_map})

data['BsmtExposure'] = data['BsmtExposure'].map(get_map_to_ordinal(['NA', 'No', 'Mn', 'Av', 'Gd']))

order = ['NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']

data['BsmtFinType1'] = data['BsmtFinType1'].map(get_map_to_ordinal(order))

data['BsmtFinType2'] = data['BsmtFinType2'].map(get_map_to_ordinal(order))

ord_vars.extend(miss_many_vars)
data[miss_many_vars + miss_few_vars].isna().sum()
data.loc[:,data.isna().any()].isna().sum()
data[['GarageCars', 'GarageArea']] = data[['GarageCars', 'GarageArea']].fillna(0)
miss_vars = ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond']

miss_5_idx = data.index[data[miss_vars].isna().all(axis=1)]

print((data.loc[miss_5_idx,'GarageArea'] == 0).sum())
for var in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:

    data[var] = data[var].fillna('NA')

data['GarageYrBlt'] = data['GarageYrBlt'].fillna(data['YearBuilt'])
data['GarageFinish'] = data['GarageFinish'].map(get_map_to_ordinal(['NA', 'Unf', 'RFn', 'Fin']))

data['GarageQual'] = data['GarageQual'].map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})

data['GarageCond'] = data['GarageCond'].map({'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5})
data['GarageFinish'].unique()
sns.boxplot(x='GarageType', y='SalePrice_log', data=data, 

            order=['NA', 'Detchd', 'CarPort', 'BuiltIn', 'Basment', 'Attchd', '2Types']);
cat_vars.append('GarageType')

ord_vars.extend(['GarageFinish', 'GarageQual', 'GarageCond'])

num_vars.extend(['GarageYrBlt', 'GarageArea', 'GarageCars'])
data.loc[:, data.isna().any()].isna().sum()
mason_vars = ['MasVnrArea', 'MasVnrType']

data.loc[data[mason_vars].isna().sum(axis=1) == 1, mason_vars]
data.loc[2611, 'MasVnrType'] = data['MasVnrType'].mode().iloc[0]

data[mason_vars] = data[mason_vars].fillna({'MasVnrArea': 0, 'MasVnrType': 'None'})
ax = sns.boxplot(x='MasVnrType', y='SalePrice_log', data=data,

                 order=['None', 'BrkCmn', 'BrkFace', 'Stone'])

data['MasVnrType'].value_counts()
data['MasVnrType'] = data['MasVnrType'].map({'None': 0, 'BrkCmn': 0, 'BrkFace': 1, 'Stone': 2})

ord_vars.append('MasVnrType')

num_vars.append('MasVnrArea')
data.loc[:, data.isna().any() > 0].isna().sum()
ext_vars = ['Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond']

data.loc[data[ext_vars].isna().any(axis=1), ext_vars]
data[ext_vars] = data[ext_vars].fillna(data[ext_vars].mode().iloc[0])
ax = sns.boxplot(x='Exterior2nd', y='SalePrice_log', data=data);

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
data['ExterQual'] = data['ExterQual'].map(qualities_map)

data['ExterCond'] = data['ExterCond'].map(qualities_map)

cat_vars.extend(['Exterior1st', 'Exterior2nd'])

ord_vars.extend(['ExterQual', 'ExterCond'])
data.loc[:, data.isna().any()].isna().sum()
len(data[(data['FireplaceQu'].isna()) & (data['Fireplaces'] == 0)])
data['FireplaceQu'] = data['FireplaceQu'].fillna('NA')

data['FireplaceQu'].unique()
data['FireplaceQu'] = data['FireplaceQu'].map(qualities_map)

data['FireplaceQu'].unique()
num_vars.append('Fireplaces')

ord_vars.append('FireplaceQu')
data.loc[:, data.isna().any()].isna().sum()
data['Electrical'] = data['Electrical'].fillna(data['Electrical'].mode().iloc[0])

data['Electrical'].value_counts()
order = ['Mix', 'FuseP', 'FuseF', 'FuseA', 'SBrkr']

sns.boxplot(x= 'Electrical', y='SalePrice_log', data=data, 

           order=order)
data['Electrical'] = data['Electrical'].map(get_map_to_ordinal(order))

sorted(data['Electrical'].unique())

ord_vars.append('Electrical')
data.loc[:, data.isna().any()].isna().sum()
data['Functional'] = data['Functional'].fillna(data['Functional'].mode().iloc[0])
order = 'Sal Sev Maj2 Maj1 Mod Min1 Min2 Typ'.split()

data['Functional'] = data['Functional'].map(get_map_to_ordinal(order))

data['Functional'].unique()

ord_vars.append('Functional')
kitchen_vars = ['KitchenAbvGr', 'KitchenQual']

data.loc[data[kitchen_vars].isna().any(axis=1), kitchen_vars]
data['KitchenQual'] = data['KitchenQual'].fillna(

    data['KitchenQual'].mode().iloc[0]).map(qualities_map)

num_vars.append('KitchenAbvGr')

ord_vars.append('KitchenQual')
data['SaleType'] = data['SaleType'].fillna(data['SaleType'].mode().iloc[0])

cat_vars.extend(['SaleType', 'SaleCondition'])
data['Utilities'].value_counts()
data = data.drop(columns=['Utilities'])
data['Alley'].value_counts(dropna=False)
data['Alley'] = data['Alley'].fillna('NA')

sns.violinplot(x='Alley', y='SalePrice_log', data=data)
cat_vars.append('Alley')
data['Fence'] = data['Fence'].fillna('NA')

sns.boxplot(x='Fence', y='SalePrice_log', data=data)
cat_vars.append('Fence')
data.loc[:, data.isna().any()].isna().sum()
both_idx = data.index[(data['PoolArea'] == 0) & (data['PoolQC'].isna())]

data.loc[both_idx, 'PoolQC'] = data.loc[both_idx, 'PoolQC'].fillna('NA')

data[data['PoolQC'].isna()]['OverallQual'] # below average, above average, fair
data.loc[2421, 'PoolQC'] = 'TA'

data.loc[2504, 'PoolQC'] = 'Gd'

data.loc[2600, 'PoolQC'] = 'Fa'
data['PoolQC'] = data['PoolQC'].map(qualities_map)
ord_vars.append('PoolQC')

num_vars.append('PoolArea')
neigh_meds = data.groupby('Neighborhood')['LotFrontage'].median()

data['LotFrontage'] = data.apply(

    lambda row: neigh_meds[row['Neighborhood']] if pd.isna(row['LotFrontage']) else row['LotFrontage'], axis=1)
num_vars.extend(['LotFrontage', 'LotArea'])
sns.boxplot(x='LotShape', y='SalePrice_log', data=data);
sns.boxplot(x='LotConfig', y='SalePrice_log', data=data);
cat_vars.extend(['LotShape', 'LotConfig'])
data['MSZoning'].value_counts(dropna=False)
neigh_modes = data.groupby('Neighborhood')['MSZoning'].agg(lambda x: x.value_counts().index[0])

data['MSZoning'] = data.apply(

    lambda row: neigh_modes[row['Neighborhood']] if pd.isna(row['MSZoning']) else row['MSZoning'], axis=1)
sns.boxplot(x='MSZoning', y='SalePrice_log', data=data);
cat_vars.append('MSZoning')
data['MiscFeature'] = data['MiscFeature'].fillna('NA')
cat_vars.append('MiscFeature')

num_vars.append('MiscVal')
print([var for var in data.select_dtypes(include=np.number).columns if var not in cat_vars+num_vars+ord_vars])
num_vars.extend(['1stFlrSF', '2ndFlrSF', '3SsnPorch', 'BedroomAbvGr', 'EnclosedPorch', 

                 'FullBath', 'GrLivArea', 'HalfBath', 'LowQualFinSF', 

                 'OpenPorchSF', 'ScreenPorch', 

                 'TotRmsAbvGrd', 'WoodDeckSF', 'YearBuilt', 'YearRemodAdd'])
print([var for var in data.select_dtypes(include=np.number).columns if var not in cat_vars+num_vars+ord_vars])
ord_vars.extend(['OverallCond', 'OverallQual'])
fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(16,6))

sns.boxplot(x='MSSubClass', y='SalePrice_log', data=data, ax=ax[0,0]);

sns.boxplot(x='MoSold', y='SalePrice_log', data=data, ax=ax[0,1]);

sns.boxplot(x='YrSold', y='SalePrice_log', data=data, ax=ax[1,0]);
cat_vars.extend(['MSSubClass','MoSold', 'YrSold'])
print([var for var in data.select_dtypes(exclude=np.number).columns if var not in cat_vars+num_vars+ord_vars])
data['CentralAir'] = data['CentralAir'].map({'Y': True, 'N': False})

data['HeatingQC'] = data['HeatingQC'].map(qualities_map)

cat_vars.extend(['CentralAir', 'Street'])

ord_vars.append('HeatingQC')
variables = [var for var in data.select_dtypes(exclude=np.number).columns 

             if var not in cat_vars+num_vars+ord_vars]

nrows = int(np.ceil(len(variables) / 3))

fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(16,24))

for i, var in enumerate(variables):

    row, col = divmod(i, 3)

    sns.boxplot(x=var, y='SalePrice_log', data=data, ax=ax[row, col])
data['LandSlope'] = data['LandSlope'].map({'Sev': 0, 'Mod': 1, 'Gtl': 2})

data['PavedDrive'] = data['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
cat_vars.extend(['BldgType', 'Condition1', 'Condition2', 'Foundation', 'Heating', 

                 'HouseStyle', 'LandContour', 'Neighborhood', 'RoofMatl', 'RoofStyle'])

ord_vars.extend(['LandSlope', 'PavedDrive'])
print([var for var in data.columns if var not in num_vars+cat_vars+ord_vars])
corr_price = abs(data.corr()['SalePrice_log']).sort_values(ascending=False) > 0.5

corr_price = corr_price[corr_price].index

plt.figure(figsize=(16,10))

sns.heatmap(data[corr_price].corr(), annot=True);
sf_vars = [var for var in data.columns if 'SF' in var] + ['GrLivArea', 'SalePrice_log']

sf_corr_order = data[sf_vars].corr()['SalePrice_log'].sort_values(ascending=False).index

plt.figure(figsize=(16,10))

sns.heatmap(data[sf_corr_order].corr(), annot=True);
indoor_sf = data[['TotalBsmtSF', 'GrLivArea']].sum(axis=1)

data['SalePrice_log'].corr(indoor_sf)
to_drop = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF', 'GrLivArea']

data = data.drop(columns=to_drop)

num_vars = [var for var in num_vars if var not in to_drop]

data['TotSF'] = indoor_sf

num_vars.append('TotSF')
sns.regplot(x='TotSF', y='SalePrice_log', data=data);
print(data[(data.index.isin(train_ids)) & (data['TotSF'] >= 7750)][['TotSF', 'SalePrice_log']])
num_baths = data['BsmtFullBath'] + data['FullBath'] + data['BsmtHalfBath']/2 + data['HalfBath']/2

data['SalePrice_log'].corr(num_baths)
data = data.drop(columns=['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath'])

num_vars = [var for var in num_vars if var not in 

            ['BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath']]

data['Bath'] = num_baths

num_vars.append('Bath')
neigh_order = data.groupby('Neighborhood')['SalePrice_log'].median().sort_values().index

fig, ax = plt.subplots(figsize=(16,6))

sns.barplot(x='Neighborhood', y='SalePrice_log', data=data, order=neigh_order, estimator=np.median, ax=ax);

plt.ylim(11,13);

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right');
rich = ['StoneBr', 'NoRidge', 'NridgHt']

poor = ['MeadowV', 'IDOTRR', 'BrDale']

def get_wealth(neigh):

    if neigh in rich:

        return 2

    if neigh in poor:

        return 0

    return 1

data['NeighWealth'] = data['Neighborhood'].apply(get_wealth)

ord_vars.append('NeighWealth')
enc_porch_vars = ['EnclosedPorch', '3SsnPorch', 'ScreenPorch']

data['EncPorch'] = data[enc_porch_vars].sum(axis=1)

data = data.drop(columns=enc_porch_vars)



op_porch_vars = ['WoodDeckSF', 'OpenPorchSF']

data['OpPorch'] = data[op_porch_vars].sum(axis=1)

data = data.drop(columns=op_porch_vars)



num_vars = [var for var in num_vars if var not in enc_porch_vars + op_porch_vars] + ['OpPorch', 'EncPorch']
data['IsRemodeled'] = data['YearBuilt'] == data['YearRemodAdd']

data['IsNew'] = data['YearBuilt'] == data['YrSold']

data['Age'] = data['YrSold'] - data['YearBuilt']

plt.figure(figsize=(16,6))

sns.regplot(data['Age'], data['SalePrice_log']);
to_drop = ['YearRemodAdd', 'YrSold', 'YearBuilt']

data = data.drop(columns=to_drop)

num_vars = [var for var in num_vars if var not in to_drop] + ['Age']

cat_vars.remove('YrSold')

cat_vars.extend(['IsRemodeled', 'IsNew'])
all_vars = num_vars + ord_vars + cat_vars

df_vars = data.columns.values

print(set(df_vars) - set(all_vars))
for var in cat_vars:

    if (data[var].value_counts() < 10).any():

        print(data[var].value_counts())

        print()
def bin_to_other(s, other, thresh=10):

    low_counts = s.value_counts() < thresh

    variables = low_counts.index[low_counts].values

    return s.apply(lambda val: other if val in variables else val)
data['Exterior1st'] = bin_to_other(data['Exterior1st'], 'Other')

data['Exterior2nd'] = bin_to_other(data['Exterior2nd'], 'Other')

data['SaleType'] = bin_to_other(data['SaleType'], 'Oth')

data['MSSubClass'] = bin_to_other(data['MSSubClass'], 'Other')

data['Condition1'] = bin_to_other(data['Condition1'], 'Other')

data['Condition2'] = bin_to_other(data['Condition2'], 'Other')

data['Heating'] = bin_to_other(data['Heating'], 'Other')

data['RoofMatl'] = bin_to_other(data['RoofMatl'], 'Other')
data['HasShed'] = data['MiscFeature'] == 'Shed'

cat_vars.append('HasShed')

data = data.drop(columns=['MiscFeature'])

cat_vars.remove('MiscFeature')
high_corr = data.corr()['SalePrice_log'].sort_values(ascending=False) > 0.5

var_high_corr = high_corr.index[high_corr]

plt.figure(figsize=(16, 10))

sns.heatmap(data[var_high_corr].corr(), annot=True);
data = data.drop(columns=['GarageArea'])

num_vars.remove('GarageArea')
to_plot = []

for var in num_vars:

    if abs(data[var].skew()) > 0.8:

        to_plot.append(var)

fig, ax = plt.subplots(nrows=len(to_plot), ncols=2, figsize=(16,6*len(to_plot)))

for row, var in enumerate(to_plot):

    skew = data[var].skew()

    log_skew = np.log1p(data[var]).skew()

    sns.distplot(data[var], kde=False, ax=ax[row,0]).set_title('Skew: {}'.format(round(skew,3)))

    sns.distplot(np.log1p(data[var]), kde=False, ax=ax[row,1]).set_title('Skew: {}'.format(round(log_skew,3)))
to_log = ['BsmtFinSF1', 'BsmtUnfSF', 'MasVnrArea', 'LotFrontage', 'LotArea', 'TotSF', 'OpPorch']

for var in to_log:

    new_var = var+'_log'

    data[new_var] = np.log1p(data[var])

    data = data.drop(columns=[var])

    num_vars.remove(var)

    num_vars.append(new_var)
data.to_csv('processed_not_scaled_no_dummies.csv')
data.shape
dummy_data = pd.get_dummies(data[cat_vars], drop_first=True)
dummy_data.columns[dummy_data.loc[test_ids].astype(bool).sum(axis=0) == 0]
dummy_data = dummy_data.drop(columns=['HouseStyle_2.5Fin'])
dummy_data.columns[dummy_data.loc[train_ids].astype(bool).sum(axis=0) < 10]
dummy_data = dummy_data.drop(columns=dummy_data.columns[dummy_data.loc[train_ids].astype(bool).sum(axis=0) < 10])
data = data.drop(columns=cat_vars).merge(dummy_data, left_index=True, right_index=True)

data.shape
data.to_csv('processed_has_dummies_not_scaled.csv')
train = data.loc[train_ids]

test = data.loc[test_ids].drop(columns=['SalePrice_log'])

print(train.shape)

print(test.shape)
train = train.drop(index=[524, 1299])
scaler = RobustScaler()

train[num_vars+ord_vars] = scaler.fit_transform(train[num_vars+ord_vars])

test[num_vars+ord_vars] = scaler.transform(test[num_vars+ord_vars])

X = train.drop(columns=['SalePrice_log'])

y = train['SalePrice_log']
lasso_cv = LassoCV().fit(X, y)

alpha_lasso = lasso_cv.alpha_
rmse_list = []

for state in range(10):

    lasso = Lasso(alpha_lasso)

    k_fold = KFold(shuffle=True, random_state=state)

    rmse_list.append(np.sqrt(-1*cross_val_score(lasso, X, y, cv=k_fold, scoring='neg_mean_squared_error')).mean())

rmse_list = np.array(rmse_list)

print(rmse_list.mean(), rmse_list.std())
ridge_cv = RidgeCV().fit(X, y)

alpha_ridge = ridge_cv.alpha_
rmse_list = []

for state in range(10):

    ridge = Ridge(alpha_ridge)

    k_fold = KFold(shuffle=True, random_state=state)

    rmse_list.append(np.sqrt(-1*cross_val_score(ridge, X, y, cv=k_fold, scoring='neg_mean_squared_error')).mean())

rmse_list = np.array(rmse_list)

print(rmse_list.mean(), rmse_list.std())
elnet_cv = ElasticNetCV().fit(X, y)

alpha_elnet = elnet_cv.alpha_
rmse_list = []

for state in range(10):

    elnet = ElasticNet(alpha_elnet)

    k_fold = KFold(shuffle=True, random_state=state)

    rmse_list.append(np.sqrt(-1*cross_val_score(elnet, X, y, cv=k_fold, scoring='neg_mean_squared_error')).mean())

rmse_list = np.array(rmse_list)

print(rmse_list.mean(), rmse_list.std())
X = train.drop(columns=['SalePrice_log'])

y = train['SalePrice_log']
xgb = XGBRegressor(objective='reg:squarederror')

k_fold = KFold(shuffle=True, random_state=0)

print(np.sqrt(-1*cross_val_score(xgb, X, y, cv=k_fold, scoring='neg_mean_squared_error')).mean())
rmse_list = []

for state in range(10):

    xgb = XGBRegressor(objective='reg:squarederror')

    k_fold = KFold(shuffle=True, random_state=state)

    rmse_list.append(np.sqrt(-1*cross_val_score(xgb, X, y, cv=k_fold, scoring='neg_mean_squared_error')).mean())

rmse_list = np.array(rmse_list)

print(rmse_list.mean(), rmse_list.std())
ridge = Ridge(alpha_ridge).fit(X, y)

train_pred = ridge.predict(X)

test_pred = ridge.predict(test)
sns.distplot(train_pred - y);
sm.qqplot(train_pred - y, fit=True, line='45');
real_scale = np.exp(test_pred) - 1

report = pd.DataFrame({'Id': test.index, 'SalePrice': real_scale})
sns.distplot(np.exp(train_pred)+1, kde=False);

sns.distplot(np.exp(test_pred)+1, kde=False);
report.to_csv('submission.csv', index=False)