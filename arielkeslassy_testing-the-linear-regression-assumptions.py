# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')
print("train", train.shape, "test", test.shape)
test.head()
train.head()
train.info()
#concat train and test to preprocces both. will split them later on BEFORE training to avoid data leakage.

X = pd.concat([train.drop("SalePrice", axis=1),test], axis=0)

y = train[['SalePrice']]

X.shape

X.info()
categorical_features = X.select_dtypes(include=object).columns.tolist()

numerical_features = X.select_dtypes(exclude=object).columns.tolist()



numerical_features
#some numerical are actually categorical

categorical_features += ['MSSubClass', 'OverallQual', 'OverallCond']

numerical_features.remove('MSSubClass')

numerical_features.remove('OverallQual')

numerical_features.remove('OverallCond')
X[categorical_features] = X[categorical_features].fillna("NA")

X
ordinal_features = ['Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure',

                    'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional', 'FireplaceQu',

                    'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'OverallQual', 'OverallCond']

binary_features = ['CentralAir']

col_list = ordinal_features + binary_features

label_features = [x for x in categorical_features if x not in col_list]

print(label_features)
#encode ordinals

for feat in ordinal_features:

    print(feat, X[feat].unique())
#creating numerical maps for encoding:

#utilities and kitchenQul will be handled separately: we need no work to replace thier "NA" values,

#because unlike the other features they are'nt equivalent to 0.

land_slope_map = {'Gtl': 3, 'Mod': 2, 'Sev': 1}

general_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}

bsmt_exposure_map = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}

bsmt_fin_types_map = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}

functional_map = {'Typ': 6, 'NA': 6, 'Min1': 5, 'Min2': 4, 'Mod': 3, 'Maj1': 2, 'Maj2': 1, 'Sev': 0}

garage_finish_map = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'NA': 0}

fence_map = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'NA': 0}
def ordinalMapper(feature, dic):

    X[feature] = X[feature].map(dic)



ordinalMapper('LandSlope', land_slope_map)

ordinalMapper('BsmtExposure', bsmt_exposure_map)

ordinalMapper('BsmtFinType1', bsmt_fin_types_map)

ordinalMapper('BsmtFinType2', bsmt_fin_types_map)

ordinalMapper('Functional', functional_map)

ordinalMapper('GarageFinish', garage_finish_map)

ordinalMapper('Fence', fence_map)



feat_list = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond','HeatingQC', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']



for feat in feat_list:

    ordinalMapper(feat, general_map)

    

X['LandSlope'].unique()
#handle kitchen and utilities

#find where there's no kitchen, according to the NUMERIC feature KitchenAbvGr

for idx, row in X.iterrows():

    if(row['KitchenAbvGr'] == 0):

        print(idx)
#mark the rows found as NA, later will turn to zeroes.

X.loc[[955, 2588, 2860], 'KitchenQual'] = 'NA'
#fill rest of nans with the feature's mode

X['KitchenQual'] = X['KitchenQual'].fillna(X['KitchenQual'].mode()[0])

print(X['KitchenQual'].unique())

print(X['KitchenQual'].value_counts())
ordinalMapper('KitchenQual', general_map)

print(X['KitchenQual'].value_counts())
X.info()
#fill utilities nans with the feature's mode

X['Utilities'] = X['Utilities'].replace('NA', X['Utilities'].mode()[0])

X['Utilities'].unique()
util_map = {'AllPub': 3, 'NoSeWa': 1}

ordinalMapper('Utilities', util_map)

X['Utilities'].unique()
#encode binary feature

X['CentralAir'].unique()
#encode binary feature

bin_map = {'Y': 1, 'N': 0}

ordinalMapper('CentralAir', bin_map)

X['CentralAir'].unique()
X.info()
X[numerical_features].info()
import seaborn as sns

import matplotlib.pyplot as plt



fig = plt.figure(figsize=(18,16))

for index,col in enumerate(numerical_features):

    plt.subplot(4,9,index+1)

    sns.distplot(X.loc[:,col], kde=False)

fig.tight_layout(pad=1.0)
#to avoid division by zero

X.replace(0, np.log(1.0001), inplace=True)
def removeRedundentFeatures(lst, categorical_features, numerical_features):

    X.drop(lst, axis=1, inplace=True)

    categorical_features = pd.Series([x for x in categorical_features if x not in lst])

    numerical_features = pd.Series([x for x in numerical_features if x not in lst])

    return categorical_features, numerical_features



redundent_features = ['BsmtFinSF2', 'LowQualFinSF', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal']

categorical_features, numerical_features = removeRedundentFeatures(redundent_features, categorical_features, numerical_features)
#remove outliers

outliers_cols = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 

                'GrLivArea', 'GarageYrBlt', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch']



fig = plt.figure(figsize=(20,7))



for idx, col in enumerate(outliers_cols):

    plt.subplot(2,6,idx+1)

    sns.boxplot(y= col, data=X)

fig.tight_layout(pad=1.5)
train.drop(train[train['LotFrontage'] > 150].index, inplace = True)

train.drop(train[train['LotArea'] > 20000].index, inplace = True)

train.drop(train[train['MasVnrArea'] > 750].index, inplace = True)

train.drop(train[train['BsmtFinSF1'] > 3000].index, inplace = True)

train.drop(train[train['TotalBsmtSF'] > 4000].index, inplace = True)

train.drop(train[train['1stFlrSF'] > 3500].index, inplace = True)

train.drop(train[train['2ndFlrSF'] > 2000].index, inplace = True)

train.drop(train[train['GrLivArea'] > 4000].index, inplace = True)

train.drop(train[train['GarageYrBlt'] > 2050].index, inplace = True)

train.drop(train[train['WoodDeckSF'] > 450].index, inplace = True)

train.drop(train[train['OpenPorchSF'] > 250].index, inplace = True)

train.drop(train[train['EnclosedPorch'] > 400].index, inplace = True)
train['LotArea'].max()
len(numerical_features)
fig = plt.figure(figsize=(18,16))

for index,col in enumerate(numerical_features):

    plt.subplot(4,7,index+1)

    sns.distplot(X.loc[:,col], kde=False)

fig.tight_layout(pad=1.0)
#remove near-single-value features

redundent_features = ['EnclosedPorch', 'OpenPorchSF', 'WoodDeckSF', 'BsmtHalfBath', '2ndFlrSF', 'BsmtFinSF1', 'MasVnrArea']

categorical_features, numerical_features = removeRedundentFeatures(redundent_features, categorical_features, numerical_features)
df_with_y = X.loc[1:1460].join(y)
df_with_y
correlations = df_with_y.corr()

correlations['SalePrice'].sort_values(ascending=False)
redundent_features = correlations['SalePrice'].loc[correlations['SalePrice'] < 0.2].index

categorical_features, numerical_features = removeRedundentFeatures(redundent_features, categorical_features, numerical_features)

numerical_features
correlations = X.corr().abs()

fig = plt.figure(figsize=(18,16))

sns.heatmap(correlations, mask = correlations < 0.8, cmap="YlGnBu")
#matrix's upper triangle to avoid duplicates

upper = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(np.bool))

upper
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]

to_drop
categorical_features, numerical_features = removeRedundentFeatures(to_drop, categorical_features, numerical_features)
X = pd.get_dummies(X)

print(X.shape)

X.head()
print((X.isna().any()).unique())

print((y.isna().any()).unique())
null_cols = X.columns[X.isna().any()].tolist()

null_cols
def fillNaWithMode(col):

    X[col].fillna(X[col].mode()[0], inplace=True)



for col in null_cols:

    fillNaWithMode(col)

print((X.isna().any()).unique())
def noTransformation(col):

    return col

def sqrtTransformation(col):

    return col**0.5

def cubeRootTransformation(col):

    return col**(1/3)

def logTransformation(col):

    return np.log(col)

def doubleLogTransformation(col):

    return np.log(np.abs(np.log(col)))

def reciprocalTransformation(col):

    return 1/col

def squareTransformation(col):

    return col**2

def thirdPowerTransformation(col):

    return col**3





def skewReduce(col):

    d = {np.abs(col.skew()): noTransformation, np.abs((col**0.5).skew()): sqrtTransformation,

         np.abs((col**(1/3)).skew()): cubeRootTransformation, np.abs((np.log(col)).skew()): logTransformation,

          np.abs((np.log(np.log(col))).skew()): doubleLogTransformation, np.abs((1/col).skew()): reciprocalTransformation, 

          np.abs((col**2).skew()): squareTransformation, np.abs((col**3).skew()): thirdPowerTransformation}

    if(np.abs(col.skew()) > 0.5):

        skew = min(d.keys())

        return d[skew]

    else:

        return noTransformation
len(numerical_features)
discrete_numerical_features = ['BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 

                              'Fireplaces', 'GarageCars', 'MoSold', 'YrSold']

continuous_numerical_features = [x for x in numerical_features if x not in discrete_numerical_features]
len(continuous_numerical_features)
for col in continuous_numerical_features:

    f = skewReduce(X[col])

    X[col] = f(X[col])

    

fig = plt.figure(figsize=(18,12))

for index,col in enumerate(continuous_numerical_features):

    plt.subplot(2,4,index+1)

    sns.distplot(X.loc[:,col], kde=False)

fig.tight_layout(pad=1.0)
sns.distplot(y['SalePrice'], kde=False)
f = skewReduce(y['SalePrice'])

y['SalePrice'] = f(y['SalePrice'])

print(f)

sns.distplot(y, kde=False)
y.head()
X.head()
X.shape, train.shape, test.shape
train_idx = [idx for idx in train.index if idx in X.index]

test_idx = [idx for idx in test.index if idx in X.index]
X_train = X.loc[train_idx]

y_train = y.loc[train_idx]

X_test = X.loc[test_idx]

print((X_train.isna().any()).unique())

X_train.shape, y_train.shape, X_test.shape
#y is double_logged

y.head()
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
y_pred.shape
preds = y_pred.flatten()

preds_inversed = np.exp(np.exp(preds))

preds_inversed
submission = pd.DataFrame({'Id': test.index,

                           'SalePrice': preds_inversed})



submission.to_csv("./submission.csv", index=False)