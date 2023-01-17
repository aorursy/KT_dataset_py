import pandas as pd

import numpy as np

import warnings

warnings.filterwarnings('ignore')



# import train and test sets



train_df = pd.read_csv('../input/train.csv', index_col=0)

test_df = pd.read_csv('../input/test.csv', index_col=0)



print('Train set shape: {}'.format(train_df.shape))

print('Test set shape: {}'.format(test_df.shape))
train_df.head()
test_df.head()
# check the number of 'number' and 'object' pandas dtypes in train and test sets



print('Number of \'number\' features in train set: {}'.format(train_df.select_dtypes(include='number').shape[1]))

print('Number of \'object\' features in train set: {}'.format(train_df.select_dtypes(include='object').shape[1]))

train_df.select_dtypes(include='number').columns, train_df.select_dtypes(include='object').columns
print('Number of \'number\' features in test set: {}'.format(test_df.select_dtypes(include='number').shape[1]))

print('Number of \'object\' features in test set: {}'.format(test_df.select_dtypes(include='object').shape[1]))

test_df.select_dtypes(include='number').columns, test_df.select_dtypes(include='object').columns
full_df = train_df.drop(columns=['SalePrice']).append(test_df)

print('Full set shape: {}'.format(full_df.shape))
features = ['MSSubClass', 'OverallQual', 'OverallCond',

            'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

            'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',

            'TotRmsAbvGrd', 'Fireplaces', 'GarageCars']

for feature in features:

    print('Feature {} has {} unique values'.format(feature, full_df[feature].nunique()))
full_df[['MSSubClass', 'OverallQual', 'OverallCond']] = full_df[['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')

train_df[['MSSubClass', 'OverallQual', 'OverallCond']] = train_df[['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')

test_df[['MSSubClass', 'OverallQual', 'OverallCond']] = test_df[['MSSubClass', 'OverallQual', 'OverallCond']].astype('object')



full_df[['MSSubClass', 'OverallQual', 'OverallCond']].dtypes
# check train and test sets distribution of numerical features and visualize standard deviations



train_num_stat = train_df.select_dtypes(include='number').describe()

train_num_stat
test_num_stat = test_df.select_dtypes(include='number').describe()

test_num_stat
import matplotlib.pyplot as plt

%matplotlib inline



std_df = pd.DataFrame(data={'Train std': train_num_stat.loc['std'].drop(['SalePrice']), 'Test std': test_num_stat.loc['std']})



plt.style.use('ggplot')

std_df.plot(kind='bar', figsize=(12, 5))

plt.title('Standard Deviations of Numerical Features', fontsize=14)



plt.show()
std_df.drop(['LotArea']).plot(kind='bar', figsize=(12, 5))

plt.title('Standard Deviations of Numerical Features', fontsize=14)



plt.show()
std_df.loc[['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

            'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

            'GarageCars']].plot(kind='bar', figsize=(8, 5))

plt.title('Standard Deviations of Numerical Features', fontsize=14)



plt.show()
train_cat_stat = train_df.select_dtypes(include='object').describe()

train_cat_stat
test_cat_stat = test_df.select_dtypes(include='object').describe()

test_cat_stat
unique_df = pd.DataFrame(data={'Train unique': train_cat_stat.loc['unique'], 'Test unique': test_cat_stat.loc['unique']})



unique_df.plot(kind='bar', figsize=(12, 5))

plt.title('Number of Unique Values of Categorical Features', fontsize=14)



plt.show()
print('Full set shape: {}'.format(full_df.shape))
# check missing features, total missing values and percentage



def check_nan(full_df):

    

    full_nan_df = full_df.isna().sum()

    full_nan_df = pd.DataFrame(data={'Missing values': full_nan_df[full_nan_df != 0], 'Percentage': round(full_nan_df[full_nan_df != 0] / full_df.shape[0], 3)})

    full_nan_df.sort_values(by='Missing values', ascending=False, inplace=True)

    nan_features = full_nan_df.index

    return nan_features,full_nan_df



nan_features,full_nan_df = check_nan(full_df)

full_nan_df
# list of features with missing values



nan_features
import seaborn as sns



features = ['Alley', 'Fence', 'Functional', 'Utilities', 'Electrical', 'KitchenQual', 'SaleType']



fig = plt.figure(figsize=(18,12))



for i, col in enumerate(features):

    fig.add_subplot(3,3,i+1)

    sns.countplot(full_df[col])



fig.add_subplot(3,3,i+2)

sns.kdeplot(full_df['LotFrontage'])



plt.show()
# drop those features from our dataset



full_df.drop(columns=['Alley', 'Fence', 'Functional', 'Utilities', 'Electrical', 'KitchenQual', 'SaleType', 'LotFrontage'], inplace=True)
misc_features = ['MiscFeature', 'MiscVal']



mask = full_df[misc_features].isna().any(axis=1)

misc_df = full_df.loc[~mask, misc_features]

misc_df.head(10)
sns.countplot(full_df['MiscFeature'])

plt.show()
sns.boxplot(x='MiscFeature', y='MiscVal', data=misc_df)



plt.show()
full_df['HasShed'] = 'No'

mask = full_df['MiscFeature'] == 'Shed'

full_df.loc[mask,'HasShed'] = 'Yes'



full_df['ShedArea'] = 0

full_df.loc[mask, 'ShedArea'] = full_df.loc[mask, 'MiscVal']



full_df.drop(columns=['MiscFeature', 'MiscVal'], inplace=True)
mask = full_df['HasShed'] != 0

full_df.loc[mask, ['HasShed', 'ShedArea']].head(10)
# look at houses with missing basement features



bsmt_features = ['BsmtExposure',

                 'BsmtCond',

                 'BsmtQual',

                 'BsmtFinType2',

                 'BsmtFinType1',

                 'BsmtFullBath',

                 'BsmtHalfBath',

                 'TotalBsmtSF',

                 'BsmtUnfSF',

                 'BsmtFinSF2',

                 'BsmtFinSF1']



mask = full_df[bsmt_features].isna().any(axis=1)

full_df.loc[mask, bsmt_features].head(10)
mask = full_df['TotalBsmtSF'] == 0

full_df.loc[mask, ['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']] = 'None'

full_df.loc[mask, ['BsmtFullBath', 'BsmtHalfBath']] = 0



mask = full_df[bsmt_features].isna().any(axis=1)

full_df.loc[mask, bsmt_features]
mask = full_df[bsmt_features].isna().all(axis=1)

full_df.loc[mask, ['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']] = 'None'

full_df.loc[mask, ['BsmtFullBath', 'BsmtHalfBath', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2', 'BsmtFinSF1']] = 0
bsmt_cat_columns = ['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinType1']

fig = plt.figure(figsize=(15,10))



for i, col in enumerate(bsmt_cat_columns):

    fig.add_subplot(2,3,i+1)

    sns.countplot(full_df[col])

    

plt.show()
columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF']

fig = plt.figure(figsize=(10, 5))



for i, col in enumerate(['BsmtFinType1', 'BsmtFinType2']):

    fig.add_subplot(1,2,i+1)

    mask = full_df[col] == 'Unf'

    data = full_df.loc[mask, columns]

    sns.boxplot(data=data)

    plt.ylabel('square feet')

    plt.title(col)



plt.show()
full_df.drop(columns=['BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType2', 'BsmtFinSF2'], inplace=True)
# let's check again missing features



nan_features,full_nan_df = check_nan(full_df)

full_nan_df
nan_features
# look at houses with missing garage features



gar_features = ['GarageYrBlt',

                'GarageCond',

                'GarageQual',

                'GarageFinish',

                'GarageType',

                'GarageCars',

                'GarageArea']



mask = full_df[gar_features].isna().any(axis=1)

full_df.loc[mask, gar_features].head(10)
full_df.drop(columns=['GarageYrBlt'], inplace=True)



mask = full_df['GarageArea'] == 0

full_df.loc[mask, ['GarageCond', 'GarageQual', 'GarageFinish', 'GarageType']] = 'None'



gar_features = ['GarageCond',

                'GarageQual',

                'GarageFinish',

                'GarageType',

                'GarageCars',

                'GarageArea']

             

mask = full_df[gar_features].isna().any(axis=1)

full_df.loc[mask, gar_features]
full_df.drop(columns=['GarageCond', 'GarageQual', 'GarageFinish'], inplace=True)



mask = full_df[['GarageCars', 'GarageArea']].isna().all(axis=1)

full_df.loc[mask, ['GarageCars', 'GarageArea']] = 0

full_df.loc[mask, ['GarageType']] = 'None'
full_df['HasBsmt'] = 'Yes'

mask = full_df['TotalBsmtSF'] == 0

full_df.loc[mask, 'HasBsmt'] = 'No'



full_df['HasGarage'] = 'Yes'

mask = full_df['GarageArea'] == 0

full_df.loc[mask, 'HasGarage'] = 'No'



full_df[['TotalBsmtSF', 'HasBsmt', 'GarageArea', 'HasGarage']].head()
# let's check again missing features



nan_features,full_nan_df = check_nan(full_df)

full_nan_df
nan_features
# look at houses with missing exterior features



exter_features = ['Exterior1st',

                  'Exterior2nd',

                  'MasVnrType',

                  'MasVnrArea',

                  'ExterQual',

                  'ExterCond']



mask = full_df[exter_features].isna().any(axis=1)

data = full_df.loc[mask, exter_features]

data.head(10)
mask = full_df['MasVnrArea'] > 0

data_mas = full_df.loc[mask, exter_features]

data_mas.head(10)
fig, axes = plt.subplots(2, 2, squeeze=True, gridspec_kw={'width_ratios': [1, 2]}, figsize=(17, 10))



a1, a2, a3, a4 = axes.flatten()



sns.countplot(x='Exterior1st', data=data, ax=a1)

a1.set_title('Masonry Veneer Missing')

sns.countplot(x='Exterior1st', hue='MasVnrType', data=data_mas, ax=a2)

a2.set_title('Masonry Veneer Not Missing')

sns.countplot(x='Exterior2nd', data=data, ax=a3)

a3.set_title('Masonry Veneer Missing')

sns.countplot(x='Exterior2nd', hue='MasVnrType', data=data_mas, ax=a4)

a4.set_title('Masonry Veneer Not Missing')



plt.show()
mask = full_df[['MasVnrType', 'MasVnrArea']].isna().all(axis=1)



full_df.loc[mask, 'MasVnrType'] = 'None'

full_df.loc[mask, 'MasVnrArea'] = 0



full_df['HasMasVnr'] = 'Yes'

mask = full_df['MasVnrArea'] == 0

full_df.loc[mask, 'HasMasVnr'] = 'No'
full_df[['MasVnrArea', 'HasMasVnr']].head()
# look again at houses with missing exterior features



mask = full_df[exter_features].isna().any(axis=1)

full_df.loc[mask, exter_features]
full_df.drop(columns=['Exterior1st', 'Exterior2nd', 'MasVnrType'], inplace=True)



nan_features,full_nan_df = check_nan(full_df)

full_nan_df
# look at houses with missing fireplace features



fp_features = ['Fireplaces', 'FireplaceQu']



mask = full_df[fp_features].isna().any(axis=1)

full_df.loc[mask, fp_features].head(10)
mask = full_df['Fireplaces'] == 0

full_df.loc[mask, 'FireplaceQu'] = 'None'



mask = full_df[fp_features].isna().any(axis=1)

full_df.loc[mask, fp_features]
# look at houses with missing pool features



pool_features = ['PoolArea', 'PoolQC']



mask = full_df[pool_features].isna().any(axis=1)

full_df.loc[mask, pool_features].head(10)
mask = full_df['PoolArea'] == 0

full_df.loc[mask, 'PoolQC'] = 'None'



mask = full_df[pool_features].isna().any(axis=1)

full_df.loc[mask, pool_features]
full_df.drop(columns=['PoolQC'], inplace=True)



full_df['HasPool'] = 'Yes'

mask = full_df['PoolArea'] == 0

full_df.loc[mask, 'HasPool'] = 'No'



full_df[['PoolArea', 'HasPool']].head()
# look at houses with missing zoning classification feature



ms_features = ['MSSubClass', 'MSZoning']



mask = full_df[ms_features].isna().any(axis=1)

data = full_df.loc[mask, ms_features]

data.head(10)
fig = plt.figure(figsize=(10, 8))



mask = full_df['MSSubClass'].isin(data['MSSubClass'].unique())

sns.countplot(x='MSSubClass', hue='MSZoning', data=full_df.loc[mask, ms_features])



plt.show()
full_df.drop(columns=['MSZoning'], inplace=True)



nan_features,full_nan_df = check_nan(full_df)

full_nan_df
train_target = train_df['SalePrice']



train_df = full_df.loc[:1460, :]

test_df = full_df.loc[1461:, :]



print('Train set shape: {}'.format(train_df.shape))

print('Test set shape: {}'.format(test_df.shape))
train_num_df = train_df.select_dtypes(include='number')

train_cat_df = train_df.select_dtypes(include='object')



print('Number of numerical features in train set: {}'.format(train_num_df.shape[1]))

print('Number of categorical features in train set: {}'.format(train_cat_df.shape[1]))



train_num_df.columns, train_cat_df.columns
features = ['YearBuilt', 'YearRemodAdd', 'MoSold', 'YrSold']



train_num_df[features].nunique().sort_values(ascending=False)
corr_features = ['LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

                 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',

                 '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea',

                 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

                 'ScreenPorch', 'PoolArea', 'ShedArea']



print(len(corr_features))
from scipy import stats



corr_df = pd.DataFrame(columns=['r', 'p-value'])



for feature in corr_features:

    corr_stats = stats.pearsonr(train_df[feature], train_target)

    corr_df.loc[feature, :] = round(corr_stats[0], 3), round(corr_stats[1], 5)



corr_df = corr_df.sort_values(by='r', ascending=False).transpose()

corr_df
corr_features = corr_df.loc['r', corr_df.loc['r'] > 0.45].index

print('Number of selected features: {}'.format(len(corr_features)))

corr_features
fig = plt.figure(figsize=(18,15))



for i, feature in enumerate(corr_features):

    fig.add_subplot(3,3,i+1)

    sns.regplot(x=train_df[feature], y=train_target)



fig.suptitle('Full Training Data Set', y=0.92, fontsize=18)



plt.show()
warnings.filterwarnings('ignore')

train_df['SalePrice'] = train_target



train_no_outl_df = train_df



masks = [(train_no_outl_df['GrLivArea'] > 4000) & (train_no_outl_df['SalePrice'] < 200000),

         (train_no_outl_df['GrLivArea'] < 5000) & (train_no_outl_df['SalePrice'] > 700000),

         (train_no_outl_df['GarageArea'] > 1200) & (train_no_outl_df['SalePrice'] < 200000),

         (train_no_outl_df['GarageArea'] < 1000) & (train_no_outl_df['SalePrice'] > 600000),

         (train_no_outl_df['TotalBsmtSF'] > 6000) & (train_no_outl_df['SalePrice'] < 200000),

         (train_no_outl_df['TotalBsmtSF'] < 3000) & (train_no_outl_df['SalePrice'] > 700000),

         (train_no_outl_df['1stFlrSF'] > 4000) & (train_no_outl_df['SalePrice'] < 200000),

         (train_no_outl_df['1stFlrSF'] < 3000) & (train_no_outl_df['SalePrice'] > 700000),

         (train_no_outl_df['MasVnrArea'] < 1200) & (train_no_outl_df['SalePrice'] > 700000)]



for mask in masks:

    train_no_outl_df = train_no_outl_df.drop(index=train_no_outl_df[mask].index)



print('Train set shape full: {}'.format(train_df.shape))

print('Train set shape outliers removed: {}'.format(train_no_outl_df.shape))
fig = plt.figure(figsize=(18,15))



for i, feature in enumerate(corr_features):

    fig.add_subplot(3,3,i+1)

    sns.regplot(x=train_no_outl_df[feature], y=train_no_outl_df['SalePrice'])

    

fig.suptitle('Outliers Removed', y=0.92, fontsize=18)



plt.show()
anova_num_features = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',

                      'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',

                      'GarageCars', 'MoSold', 'YrSold']



print('Number of discrete numerical features selected for ANOVA: {}'.format(len(anova_num_features)))
anova_num_df = pd.DataFrame(columns=['F_value', 'p-value'])



for feature in anova_num_features:

    samples = train_df[[feature, 'SalePrice']].groupby(feature)['SalePrice'].apply(list)

    anova_stats = stats.f_oneway(*samples)

    anova_num_df.loc[feature, :] = round(anova_stats[0], 3), round(anova_stats[1], 5)



anova_num_df = anova_num_df.sort_values(by='F_value', ascending=False).transpose()

anova_num_df
anova_num_features = anova_num_df.loc['F_value', anova_num_df.loc['F_value'] > 1].index

print('Number of selected features for further analysis: {}'.format(len(anova_num_features)))

anova_num_features
fig = plt.figure(figsize=(18,15))



for i, feature in enumerate(anova_num_features):

    fig.add_subplot(3,3,i+1)

    order = train_df[[feature, 'SalePrice']].groupby(feature).mean().sort_values(by='SalePrice').index

    sns.boxplot(x=train_df[feature], y=train_df['SalePrice'], order=order)



fig.suptitle('Full Training Data Set', y=0.92, fontsize=18)



plt.show()
selected_features = ['GarageCars', 'FullBath', 'Fireplaces', 'HalfBath']



mask_f50 = anova_num_df[selected_features].loc['F_value'] > 50

mask_f100 = anova_num_df[selected_features].loc['F_value'] > 100



anova_num_features_f50 = anova_num_df[selected_features].loc['F_value', mask_f50].index

anova_num_features_f100 = anova_num_df[selected_features].loc['F_value', mask_f100].index
print('Selected features with F-value > 50: {}'.format(len(anova_num_features_f50)))

print(anova_num_features_f50.tolist())

print()

print('Selected features with F-value > 100: {}'.format(len(anova_num_features_f100)))

print(anova_num_features_f100.tolist())
anova_cat_df = pd.DataFrame(columns=['F_value', 'p-value'])



for feature in train_cat_df.columns:

    samples = train_df[[feature, 'SalePrice']].groupby(feature)['SalePrice'].apply(list)

    anova_stats = stats.f_oneway(*samples)

    anova_cat_df.loc[feature, :] = round(anova_stats[0], 3), round(anova_stats[1], 5)



anova_cat_df = anova_cat_df.sort_values(by='F_value', ascending=False).transpose()

anova_cat_df
anova_cat_features = anova_cat_df.loc['F_value', anova_cat_df.loc['F_value'] > 50].index

print('Number of selected features for further analysis: {}'.format(len(anova_cat_features)))

anova_cat_features
fig = plt.figure(figsize=(18,23))



for i, feature in enumerate(anova_cat_features):

    fig.add_subplot(4,3,i+1)

    order = train_df[[feature, 'SalePrice']].groupby(feature).mean().sort_values(by='SalePrice').index

    sns.boxplot(x=train_df[feature], y=train_df['SalePrice'], order=order)

    plt.xticks(rotation=90)



plt.show()
anova_cat_features_f50 = anova_cat_features

anova_cat_features_f100 = anova_cat_df.loc['F_value', anova_cat_df.loc['F_value'] > 100].index
print('Selected features with F-value > 50: {}'.format(len(anova_cat_features_f50)))

print(anova_cat_features_f50.tolist())

print()

print('Selected features with F-value > 100: {}'.format(len(anova_cat_features_f100)))

print(anova_cat_features_f100.tolist())
print('Number of selected numerical features: {}'.format(len(corr_features)))

print(corr_features.tolist())

print()

print('Number of selected discrete numerical features with F-value > 50: {}'.format(len(anova_num_features_f50)))

print(anova_num_features_f50.tolist())

print('Number of selected discrete numerical features with F-value > 100: {}'.format(len(anova_num_features_f100)))

print(anova_num_features_f100.tolist())

print()

print('Number of selected categorical features with F-value > 50: {}'.format(len(anova_cat_features_f50)))

print(anova_cat_features_f50.tolist())

print('Number of selected categorical features with F-value > 50: {}'.format(len(anova_cat_features_f100)))

print(anova_cat_features_f100.tolist())
train_df[anova_num_features_f50] = train_df[anova_num_features_f50].astype('object')

train_no_outl_df[anova_num_features_f50] = train_no_outl_df[anova_num_features_f50].astype('object')

test_df[anova_num_features_f50] = test_df[anova_num_features_f50].astype('object')
print(train_df[anova_num_features_f50].dtypes)
ohe_features_f50 = list(anova_num_features_f50) + list(anova_cat_features_f50)



print('Number of unique values of categorical features (F-value > 50) in train set: {}'.format(train_df[ohe_features_f50].nunique().sum()))

print('Number of unique values of categorical features (F-value > 50) in train set (outliers removed): {}'.format(train_no_outl_df[ohe_features_f50].nunique().sum()))

print('Number of unique values of categorical features ((F-value > 50) in test set: {}'.format(test_df[ohe_features_f50].nunique().sum()))



ohe_features_f100 = list(anova_num_features_f100 ) + list(anova_cat_features_f100)



print()

print('Number of unique values of categorical features (F-value > 100) in train set: {}'.format(train_df[ohe_features_f100].nunique().sum()))

print('Number of unique values of categorical features (F-value > 100) in train set (outliers removed): {}'.format(train_no_outl_df[ohe_features_f100].nunique().sum()))

print('Number of unique values of categorical features ((F-value > 100) in test set: {}'.format(test_df[ohe_features_f100 ].nunique().sum()))
from sklearn.preprocessing import OneHotEncoder



ohe = OneHotEncoder(handle_unknown='ignore')

X_train_ohe_f50 = ohe.fit_transform(train_df[ohe_features_f50]).toarray()

X_train_no_outl_ohe_f50 = ohe.fit_transform(train_no_outl_df[ohe_features_f50]).toarray()

X_test_ohe_f50 = ohe.transform(test_df[ohe_features_f50]).toarray()



X_train_ohe_f100 = ohe.fit_transform(train_df[ohe_features_f100]).toarray()

X_train_no_outl_ohe_f100 = ohe.fit_transform(train_no_outl_df[ohe_features_f100]).toarray()

X_test_ohe_f100 = ohe.transform(test_df[ohe_features_f100]).toarray()
print(X_train_ohe_f50.shape)

print(X_train_no_outl_ohe_f50.shape)

print(X_test_ohe_f50.shape)

print()

print(X_train_ohe_f100.shape)

print(X_train_no_outl_ohe_f100.shape)

print(X_test_ohe_f100.shape)
# concatenate one-hot encoded features array and numerical features array



X_train_num = train_df[corr_features].get_values()

X_train_no_outl_num = train_no_outl_df[corr_features].get_values()

X_test_num = test_df[corr_features].get_values()



X_train_raw_f50 = np.concatenate((X_train_ohe_f50, X_train_num), axis=1)

X_train_no_outl_raw_f50 = np.concatenate((X_train_no_outl_ohe_f50, X_train_no_outl_num), axis=1)

X_test_raw_f50 = np.concatenate((X_test_ohe_f50, X_test_num), axis=1)



X_train_raw_f100 = np.concatenate((X_train_ohe_f100, X_train_num), axis=1)

X_train_no_outl_raw_f100 = np.concatenate((X_train_no_outl_ohe_f100, X_train_no_outl_num), axis=1)

X_test_raw_f100 = np.concatenate((X_test_ohe_f100, X_test_num), axis=1)
np.set_printoptions(suppress=True)

X_train_raw_f50
X_train_no_outl_raw_f100
print(X_train_raw_f50.shape)

print(X_train_no_outl_raw_f50.shape)

print(X_test_raw_f50.shape)

print()

print(X_train_raw_f100.shape)

print(X_train_no_outl_raw_f100.shape)

print(X_test_raw_f100.shape)
X_train_raw_f50.std(axis=0)
X_train_raw_f50.mean(axis=0)
# implement standardization either only numeric or all features



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



# standardize all features with and without outliers

def standardize_all(X_train_raw, X_test_raw):

    

    X_train_scaled_all = scaler.fit_transform(X_train_raw)

    X_test_scaled_all = scaler.transform(X_test_raw)

    return X_train_scaled_all, X_test_scaled_all



X_train_scaled_all_f50, X_test_scaled_all_f50 = standardize_all(X_train_raw_f50, X_test_raw_f50)

X_train_scaled_all_f100, X_test_scaled_all_f100 = standardize_all(X_train_raw_f100, X_test_raw_f100)

    

X_train_no_outl_scaled_all_f50, X_test_no_outl_scaled_all_f50 = standardize_all(X_train_no_outl_raw_f50, X_test_raw_f50)

X_train_no_outl_scaled_all_f100, X_test_no_outl_scaled_all_f100 = standardize_all(X_train_no_outl_raw_f100, X_test_raw_f100)

   

# standardize numerical features with and without outliers

def standardize_num(X_train_num, X_test_num):

    

    X_train_num_scaled = scaler.fit_transform(X_train_num)

    X_test_num_scaled = scaler.transform(X_test_num)

    return X_train_num_scaled, X_test_num_scaled



X_train_num_scaled, X_test_num_scaled = standardize_num(X_train_num, X_test_num)

X_train_no_outl_num_scaled, X_test_no_outl_num_scaled = standardize_num(X_train_no_outl_num, X_test_num)



# concatenate not standardized one-hot encoded features with standardized numerical

X_train_scaled_num_f50 = np.concatenate((X_train_ohe_f50, X_train_num_scaled), axis=1)

X_test_scaled_num_f50 = np.concatenate((X_test_ohe_f50, X_test_num_scaled), axis=1)

X_train_scaled_num_f100 = np.concatenate((X_train_ohe_f100, X_train_num_scaled), axis=1)

X_test_scaled_num_f100 = np.concatenate((X_test_ohe_f100, X_test_num_scaled), axis=1)



X_train_no_outl_scaled_num_f50 = np.concatenate((X_train_no_outl_ohe_f50, X_train_no_outl_num_scaled), axis=1)

X_test_no_outl_scaled_num_f50 = np.concatenate((X_test_ohe_f50, X_test_no_outl_num_scaled), axis=1)

X_train_no_outl_scaled_num_f100 = np.concatenate((X_train_no_outl_ohe_f100, X_train_no_outl_num_scaled), axis=1)

X_test_no_outl_scaled_num_f100 = np.concatenate((X_test_ohe_f100, X_test_no_outl_num_scaled), axis=1)
X_train_scaled_num_f50.std(axis=0)
X_train_scaled_num_f50.mean(axis=0)
X_train_scaled_all_f50.std(axis=0)
X_train_scaled_all_f50.mean(axis=0)
data = [X_train_num,

        X_train_num_scaled,

        X_train_raw_f100,

        X_train_scaled_num_f100,

        X_train_scaled_all_f100,

        X_train_raw_f50,

        X_train_scaled_num_f50,

        X_train_scaled_all_f50]



data_no_outl = [X_train_no_outl_num,

                X_train_no_outl_num_scaled,

                X_train_no_outl_raw_f100,

                X_train_no_outl_scaled_num_f100,

                X_train_no_outl_scaled_all_f100,

                X_train_no_outl_raw_f50,

                X_train_no_outl_scaled_num_f50,

                X_train_no_outl_scaled_all_f50]



for X in data:

    print(X.shape)
for X in data_no_outl:

    print(X.shape)
y_train = train_df['SalePrice'].get_values()

y_train_no_outl = train_no_outl_df['SalePrice'].get_values()



fig = plt.figure(figsize=(15, 5))



fig.add_subplot(1,2,1)

sns.distplot(y_train)

plt.title('Full Data Set')



fig.add_subplot(1,2,2)

sns.distplot(y_train_no_outl)

plt.title('Outliers Removed')



fig.suptitle('Distribution of Target not Transformed', y=0.999, fontsize=16)



print('The skewness of target not transformed:')

print('\tFull data set: {}'.format(stats.skew(y_train)))

print('\tOutliers removed: {}'.format(stats.skew(y_train_no_outl)))
y_train_log = np.log(train_df['SalePrice'].get_values())

y_train_no_outl_log = np.log(train_no_outl_df['SalePrice'].get_values())



fig = plt.figure(figsize=(15, 5))



fig.add_subplot(1,2,1)

sns.distplot(y_train_log)

plt.title('Full Data Set')



fig.add_subplot(1,2,2)

sns.distplot(y_train_no_outl_log)

plt.title('Outliers Removed')



fig.suptitle('Distribution of Target Log Transformed', y=0.999, fontsize=16)



print('The skewness of target log transformed:')

print('\tFull data set: {}'.format(stats.skew(y_train_log)))

print('\tOutliers removed: {}'.format(stats.skew(y_train_no_outl_log)))
from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer, mean_absolute_error



columns = [('NumOnlyTrain', 'NumOnlyVal'),

           ('NumOnlyStdTrain', 'NumOnlyStdVal'),

           ('F100Train', 'F100Val'),

           ('F100StdNumTrain', 'F100StdNumVal'),

           ('F100StdAllTrain', 'F100StdAllVal'),

           ('F50Train', 'F50Val'),

           ('F50StdNumTrain', 'F50StdNumVal'),

           ('F50StdAllTrain', 'F50StdAllVal')]           



pd.options.display.float_format = '{:,.0f}'.format



# define custom scoring function for cross validation to compare log-transformed and not transformed mae

def custom_scorer(y_true, y_pred):

    return np.mean(np.abs(np.exp(y_true) - np.exp(y_pred)))



score = make_scorer(custom_scorer, greater_is_better=False)



def model_eval(models, data, y, cv=5, transform=False):

    

    results_df = pd.DataFrame()

    for model in models:

        for X, col in zip(data, columns):

            if transform:

                scores = cross_validate(models[model], X, y, scoring=score, cv=cv, return_train_score=True)

            else:

                scores = cross_validate(models[model], X, y, scoring='neg_mean_absolute_error', cv=cv, return_train_score=True)

            results_df.loc[model, col[0]] = -round(scores['train_score'].mean(), 0)

            results_df.loc[model, col[1]] = -round(scores['test_score'].mean(), 0)

        print('{} model validation complete'.format(model))

    return results_df



def min_results(full_df, no_outl_df, log_df, no_outl_log_df):

    

    df = pd.DataFrame(data={'Min MAE Full': full_df.min(),

                            'Models Min MAE Full': full_df.idxmin(),

                            'Min MAE No Outlrs': no_outl_df.min(),

                            'Models Min MAE No Outlrs': no_outl_df.idxmin(),

                            'Min MAE Full Log Trans': log_df.min(),

                            'Models Min MAE Full Log Trans': log_df.idxmin(),

                            'Min MAE No Outlrs Log Trans': no_outl_log_df.min(),

                            'Models Min MAE No Outlrs Log Trans': no_outl_log_df.idxmin()})



    df_train = df[df.index.str.contains('Train')]

    df_val = df[df.index.str.contains('Val')]    

    return df_train, df_val
from sklearn.linear_model import ARDRegression, BayesianRidge, ElasticNet, HuberRegressor, Lars

from sklearn.linear_model import Lasso, LassoLars, LinearRegression, PassiveAggressiveRegressor, RANSACRegressor

from sklearn.linear_model import Ridge, SGDRegressor, TheilSenRegressor  



models = {'Bayesian Ridge Regression': BayesianRidge(),

          'ElasticNet Regression': ElasticNet(random_state=0),

          'Huber Regressor': HuberRegressor(),

          'Lasso Regression': Lasso(random_state=0),

          'Lasso Lars Regression': LassoLars(),

          'Linear Regression': LinearRegression(),

          'Passive Aggressive Regressor': PassiveAggressiveRegressor(random_state=0),

          'RANSAC Regressor': RANSACRegressor(base_estimator=HuberRegressor(), random_state=0),

          'Ridge Regressor': Ridge(),

          'SGD Regressor': SGDRegressor(random_state=0),

          'Theil-Sen Estimator': TheilSenRegressor(random_state=0)}



results_lm_df = model_eval(models, data, y_train)

results_lm_df
results_lm_no_outl_df = model_eval(models, data_no_outl, y_train_no_outl)

results_lm_no_outl_df
results_lm_log_df = model_eval(models, data, y_train_log, transform=True)

results_lm_log_df
results_lm_no_outl_log_df = model_eval(models, data_no_outl, y_train_no_outl_log, transform=True)

results_lm_no_outl_log_df
min_lm_train, min_lm_val = min_results(results_lm_df, results_lm_no_outl_df, results_lm_log_df, results_lm_no_outl_log_df)

print(min_lm_train.min(axis=1).sort_values())

min_lm_train
print(min_lm_val.min(axis=1).sort_values())

min_lm_val
from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor



models = {'AdaBoost Regressor': AdaBoostRegressor(random_state=0),

          'Bagging Regressor': BaggingRegressor(random_state=0),

          'ExtraTrees Regressor': ExtraTreesRegressor(random_state=0),

          'GradientBoosting Regressor': GradientBoostingRegressor(random_state=0),

          'RandomForest Regressor': RandomForestRegressor(random_state=0)}

          

results_em_df = model_eval(models, data, y_train)

results_em_df
results_em_no_outl_df = model_eval(models, data_no_outl, y_train_no_outl)

results_em_no_outl_df
results_em_log_df = model_eval(models, data, y_train_log, transform=True)

results_em_log_df
results_em_no_outl_log_df = model_eval(models, data_no_outl, y_train_no_outl_log, transform=True)

results_em_no_outl_log_df
min_em_train, min_em_val = min_results(results_em_df, results_em_no_outl_df, results_em_log_df, results_em_no_outl_log_df)

print(min_em_train.min(axis=1).sort_values())

min_em_train
print(min_em_val.min(axis=1).sort_values())

min_em_val
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor



models = {'DecisionTree Regressor': DecisionTreeRegressor(random_state=0),

          'ExtraTree Regressor': ExtraTreeRegressor(random_state=0)}

          

results_tr_df = model_eval(models, data, y_train)

results_tr_df
results_tr_no_outl_df = model_eval(models, data_no_outl, y_train_no_outl)

results_tr_no_outl_df
results_tr_log_df = model_eval(models, data, y_train_log, transform=True)

results_tr_log_df
results_tr_no_outl_log_df = model_eval(models, data_no_outl, y_train_no_outl_log, transform=True)

results_tr_no_outl_log_df
min_tr_train, min_tr_val = min_results(results_tr_df, results_tr_no_outl_df, results_tr_log_df, results_tr_no_outl_log_df)

print(min_tr_train.min(axis=1).sort_values())

min_tr_train
print(min_tr_val.min(axis=1).sort_values())

min_tr_val
from sklearn.kernel_ridge import KernelRidge

from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor





models = {'KernelRidge Regression': KernelRidge(),

          'KNeighbors Regressor': KNeighborsRegressor(),

          'RadiusNeighbors Regressor': RadiusNeighborsRegressor()}

          

results_oth_df = model_eval(models, data, y_train)

results_oth_df
results_oth_no_outl_df = model_eval(models, data_no_outl, y_train_no_outl)

results_oth_no_outl_df
results_oth_log_df = model_eval(models, data, y_train_log, transform=True)

results_oth_log_df
results_oth_no_outl_log_df = model_eval(models, data_no_outl, y_train_no_outl_log, transform=True)

results_oth_no_outl_log_df
min_oth_train, min_oth_val = min_results(results_oth_df, results_oth_no_outl_df, results_oth_log_df, results_oth_no_outl_log_df)

print(min_oth_train.min(axis=1).sort_values())

min_oth_train
print(min_oth_val.min(axis=1).sort_values())

min_oth_val
from xgboost import XGBRegressor



models = {'XGBoost Regressor': XGBRegressor(n_estimators=300, random_state=0)}

          

results_xgb_df = model_eval(models, data, y_train)

results_xgb_df
results_xgb_no_outl_df = model_eval(models, data_no_outl, y_train_no_outl)

results_xgb_no_outl_df
results_xgb_log_df = model_eval(models, data, y_train_log, transform=True)

results_xgb_log_df
results_xgb_no_outl_log_df = model_eval(models, data_no_outl, y_train_no_outl_log, transform=True)

results_xgb_no_outl_log_df
min_xgb_train, min_xgb_val = min_results(results_xgb_df, results_xgb_no_outl_df, results_xgb_log_df, results_xgb_no_outl_log_df)

print(min_xgb_train.min(axis=1).sort_values())

min_xgb_train[['Min MAE Full', 'Min MAE No Outlrs', 'Min MAE Full Log Trans', 'Min MAE No Outlrs Log Trans']]
print(min_xgb_val.min(axis=1).sort_values())

min_xgb_val[['Min MAE Full', 'Min MAE No Outlrs', 'Min MAE Full Log Trans', 'Min MAE No Outlrs Log Trans']]