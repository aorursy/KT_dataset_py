import os

import pandas as pd



import numpy as np



from sklearn.linear_model import Ridge, Lasso, LinearRegression

from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.ensemble import GradientBoostingRegressor

from mlxtend.regressor import StackingCVRegressor

from sklearn.svm import SVR

from sklearn.preprocessing import RobustScaler

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax, zscore

from multiprocessing import cpu_count

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor

import matplotlib.pyplot as plt

import seaborn as sns
np.random.seed = 42
data_dir = '/kaggle/input/house-prices-advanced-regression-techniques/'



p_train = pd.read_csv(os.path.join(data_dir, "train.csv"), index_col=0)

p_test = pd.read_csv(os.path.join(data_dir, "test.csv"), index_col=0)

p_sample = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
# fig, ax = plt.subplots(figsize=(30,10))

sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', data=p_train, palette='RdBu')
# Numerical cols

cols = list(X_train.select_dtypes('number').columns)

fig, axs = plt.subplots(int(np.sqrt(len(cols))), int(np.sqrt(len(cols))), figsize=(30, 30))

for col, ax in zip(cols, axs.ravel()):

    sns.regplot(x=X_train[col], y=y_train, ax=ax);
p_train
from scipy import stats



# # Remove outliers 

p_train.drop(p_train[(p_train['OverallQual']<5) & (p_train['SalePrice']>200000)].index, inplace=True)

p_train.drop(p_train[(p_train['GrLivArea']>4500) & (p_train['SalePrice']<300000)].index, inplace=True)

p_train.reset_index(drop=True, inplace=True)
cols = ["MSZoning"] 



# a = []

# for n in X.Neighborhood.unique():

#     X[cols].mode().iloc[0]

#     a.app

# X[cols].mode().iloc[0]

p_train.groupby(['MSSubClass'])[cols].agg(pd.Series.mode).head(10)
cols = ["Exterior1st", "Exterior2nd", "SaleType", "Electrical", "KitchenQual", "Functional"] 



# a = []

# for n in X.Neighborhood.unique():

#     X[cols].mode().iloc[0]

#     a.app

# X[cols].mode().iloc[0]

p_train.groupby(['Neighborhood'])[cols].agg(pd.Series.mode).head(5)
# Train + Test features

X_train = p_train.drop("SalePrice", axis=1)

X_test = p_test

X = pd.concat([X_train, X_test])



# Get labels (logarithmic due to distribution)

y_train = np.log1p(p_train.loc[:,"SalePrice"])
# No variance

X = X.drop(['Utilities'], axis=1)
X['MasVnrType'].value_counts()
# Fill PoolQC where it should be

mask = (X['PoolQC'].isna() & (X['PoolArea'] > 0))

X.loc[mask, ['PoolQC', 'PoolArea', 'OverallQual']] # Filling 3 rows based on overall quality of the house (assuming by intuition)
# Fill NaNs with modes for basement columns where basement data is partially missing

# We don't fill for fully missing because that means no basement

cols = ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']

temp = (X[cols].isna().sum(axis=1))

mask = (temp > 0) & (temp < 5)

X.loc[mask, cols] = X.loc[mask, cols].fillna(X.mode().iloc[0])

X.mode()[cols]

X.loc[mask, cols]
# Fill MasVnrType with non-na mode if MasVnrArea is not NaN (1 sample fixed)

mask = X['MasVnrType'].isna() & X['MasVnrArea'].notna()

mode = X.loc[X['MasVnrType'] != 'None','MasVnrType'].mode()[0]

X.loc[mask, 'MasVnrType'] = X.loc[mask, 'MasVnrType'].fillna(mode)

print('Filled with ' + mode)
# Impute MSZoning values based on modes in each MSSubClass.

X['MSZoning'] = X.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))
# Impute missing values



# Categorical (big number of nans (79+))

# NaNs here indicate lack of something (no pool, no basement, etc)

cols = ["PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageCond", "GarageQual", "GarageFinish", "GarageType", "BsmtCond", "BsmtExposure", "BsmtQual", "BsmtFinType2", "BsmtFinType1", "MasVnrType"]

X[cols] = X[cols].fillna("None")



# Impute using Neighborhoos mode (small numbers of NaNs)

cols = ["MSZoning", "Exterior1st", "Exterior2nd", "SaleType", "Electrical", "KitchenQual", "Functional"]

X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.mode()[0]))



# Impute using Neighborhoods median

cols = ["GarageArea", "LotFrontage"]

X[cols] = X.groupby("Neighborhood")[cols].transform(lambda x: x.fillna(x.median()))



# Numerical

cols = ["GarageYrBlt", "MasVnrArea", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "GarageCars"]

X[cols] = X[cols].fillna(0)
assert X.isna().sum().sum() == 0
ordinalised_cols = []
fig, axs = plt.subplots(int(np.floor(np.sqrt(len(cols)))), int(np.ceil(np.sqrt(len(cols)))), figsize=(25,15))

cols = [

 'ExterQual',

#  'ExterCond',

#  'BsmtQual',

 'BsmtCond',

 'HeatingQC',

 'KitchenQual',

#  'FireplaceQu',

 'GarageQual',

 'GarageCond',

 'PoolQC',

]



order = ['Po', 'Fa', 'TA', 'Gd', 'Ex']

for col, ax in zip(cols, np.hstack(axs)):

    print(col, X[col].unique() ,np.unique(pd.Categorical(X[col], categories=order, ordered=True).codes))

#     print(X[col].value_counts())

    X[col+'_int'] = pd.Categorical(X[col], categories=order, ordered=True).codes

    ordinalised_cols.append(col)

    print(list(X[col].value_counts()))

    sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'Po', 'Fa', 'TA', 'Gd', 'Ex'], inner=None, ax=ax)

#     sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None'] + order, ax=ax)

#     sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None'] + order, ax=ax)

#     sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None'] + order, ax=ax)

# Looks good

col = 'LotShape'

order=['Reg', 'IR1', 'IR2', 'IR3']

print(X[col].value_counts())

X[col + '_int'] = pd.Categorical(X[col], categories=order, ordered=True).codes

ordinalised_cols.append(col)

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), hue=X[col], order=order)

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=order)

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=order)

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=order)
# Does not look that informative

col = 'LandSlope'

X[col + '_int'] = pd.Categorical(X[col], categories=['Gtl', 'Mod', 'Sev'], ordered=True).codes

ordinalised_cols.append(col)

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Gtl', 'Mod', 'Sev'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Gtl', 'Mod', 'Sev'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Gtl', 'Mod', 'Sev'])
# Looks good

# col = 'GarageFinish'

# X[col + '_int'] = pd.Categorical(X[col], categories=['Unf', 'RFn', 'Fin'], ordered=True).codes

# ordinalised_cols.append(col)

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'Unf', 'RFn', 'Fin'])
# Looks good. -1 seems a much lower than others though

col = 'BsmtExposure'

X[col + '_int'] = pd.Categorical(X[col], categories=['No', 'Mn', 'Av', 'Gd'], ordered=True).codes

ordinalised_cols.append(col)

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'No', 'Mn', 'Av', 'Gd'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'No', 'Mn', 'Av', 'Gd'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'No', 'Mn', 'Av', 'Gd'])

X[['BsmtExposure', 'BsmtExposure_int']].sample(11)
# Looks all over place

col = 'Functional'

X[col + '_int'] = pd.Categorical(X[col], categories=['Min2', 'Min1', 'Typ'], ordered=True).codes # ???

print(X[col].value_counts())

X[col + '_int'] = pd.Categorical(X[col], categories=['Typ','Min1','Min2','Maj1','Maj2','Mod', 'Sev'], ordered=True).codes # ???

X[col].unique().tolist(), X[col + '_int'].unique().tolist()

# ordinalised_cols.append(col)

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Typ','Min1','Min2','Maj1','Maj2','Mod', 'Sev'])

# sns.violinplot(x=col+'_int', y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1))

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Typ','Min1','Min2','Maj1','Maj2','Mod', 'Sev'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Typ','Min1','Min2','Maj1','Maj2','Mod', 'Sev'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Typ','Min1','Min2','Maj1','Maj2','Mod', 'Sev'])
# # Shit according to FI (looks all over place)

col = 'HouseStyle'

X[col + '_int'] = pd.Categorical(X[col], categories=['2.5Fin','2Story','1Story','SLvl','2.5Unf','1.5Fin', 'SFoyer', '1.5Unf'], ordered=True).codes # ???

X['HouseStyle_1st'] = 1*(X['HouseStyle'] == '1Story')

X['HouseStyle_2st'] = 1*(X['HouseStyle'] == '2Story')

X['HouseStyle_15st'] = 1*(X['HouseStyle'] == '1.5Fin')

ordinalised_cols.append(col)

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['2.5Fin','2Story','1Story','SLvl','2.5Unf','1.5Fin', 'SFoyer', '1.5Unf'])

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['2.5Fin','2Story','1Story','SLvl','2.5Unf','1.5Fin', 'SFoyer', '1.5Unf'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['2.5Fin','2Story','1Story','SLvl','2.5Unf','1.5Fin', 'SFoyer', '1.5Unf'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['2.5Fin','2Story','1Story','SLvl','2.5Unf','1.5Fin', 'SFoyer', '1.5Unf'])
col = 'Foundation'

X[col + '_int'] = pd.Categorical(X[col], categories=['PConc', 'CBlock', 'BrkTil'], ordered=True).codes # to int

X[col + '_int'] = X[col + '_int'].replace(-1, X['Foundation_int'].max() + 1) # What's not defined before is highest

X[col].unique(), X[col + '_int'].unique()

ordinalised_cols.append(col)

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=["PConc", 'Wood',  "CBlock",'Stone', 'BrkTil', 'Slab'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=["PConc", 'Wood',  "CBlock",'Stone', 'BrkTil', 'Slab'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=["PConc", 'Wood',  "CBlock",'Stone', 'BrkTil', 'Slab'])

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=["PConc", 'Wood',  "CBlock",'Stone', 'BrkTil', 'Slab'])
# Masonry

# Looks ordinal if we combine BrkCmn and None into one (since they have the same distributions)

col = 'MasVnrType'

X[col + '_int'] = pd.Categorical(X[col].replace('BrkCmn', 'BrkCmn/None').replace('None','BrkCmn/None'),

                                 categories=['BrkCmn/None', 'BrkFace', 'Stone'], ordered=True).codes # 'Stone', 'BrkFace',  'BrkCmn', 'None'

ordinalised_cols.append(col)

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'BrkCmn', 'BrkFace',  'Stone'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'BrkCmn', 'BrkFace',  'Stone'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'BrkCmn', 'BrkFace',  'Stone'])

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'BrkCmn', 'BrkFace',  'Stone'])
# col = 'BsmtFinType1'

# X[col + '_int'] = pd.Categorical(X[col], categories=['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'], ordered=True).codes

# X[col + '_int'] = X[col + '_int'].replace(-1, X[col + '_int'].max() + 1) # What's not defined before is highest

# X[col + '_Unf'] = 1*(X[col] == 'Unf')

# ordinalised_cols.append(col)

# X[col].unique(), X[col + '_int'].unique()

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'])
# col = 'BsmtFinType2'

# cats = ['None', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ']

# X[col + '_int'] = pd.Categorical(X[col], categories=cats, ordered=True).codes

# X[col + '_int'] = X[col + '_int'].replace(-1, X[col + '_int'].max() + 1) # What's not defined before is highest

# X[col + '_Unf'] = 1*(X[col] == 'Unf')

# ordinalised_cols.append(col)

# X[col].unique(), X[col + '_int'].unique()

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=cats)
# Looks good

col = 'PavedDrive'

X[col + '_int'] = pd.Categorical(X[col], categories=['N', 'P', 'Y'], ordered=True).codes

ordinalised_cols.append(col)

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['N', 'P', 'Y'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['N', 'P', 'Y'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['N', 'P', 'Y'])

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['N', 'P', 'Y'])
# # Looks good

# col = 'CentralAir'

# X[col + '_int'] = pd.Categorical(X[col], categories=['N', 'Y'], ordered=True).codes

# ordinalised_cols.append(col)

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['N', 'Y'])
# Looks good

col = 'Street'

X[col + '_int'] = pd.Categorical(X[col], categories=['Grvl', 'Pave'], ordered=True).codes

ordinalised_cols.append(col)

sns.boxplot(x=col, y="SalePrice",  color="1", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Grvl', 'Pave'])

sns.stripplot(x=col, y="SalePrice", size=1.5, color="red", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Grvl', 'Pave'])

sns.pointplot(x=col, y="SalePrice", color="0", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Grvl', 'Pave'])

# sns.violinplot(x=col, y="SalePrice", data=pd.concat([X.loc[p_train.index], y_train], axis=1), order=['Grvl', 'Pave'])
# X['HasWoodDeck'] = (X['WoodDeckSF'] > 0) * 1

# X['HasOpenPorch'] = (X['OpenPorchSF'] > 0) * 1

# X['HasEnclosedPorch'] = (X['EnclosedPorch'] > 0) * 1

# X['Has3SsnPorch'] = (X['3SsnPorch'] > 0) * 1

# X['HasScreenPorch'] = (X['ScreenPorch'] > 0) * 1

X['YearsSinceRemod'] = X['YrSold'].astype(int) - X['YearRemodAdd'].astype(int)

# X['OverallQualCond'] = X['OverallQual'] + X['OverallCond']
# Total square footage (total, porch, bath) FE

X["TotalSF"] = X["GrLivArea"] + X["TotalBsmtSF"] # Total square footage

X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"] # Total porch square footage

X["TotalBath"] = X["FullBath"] + X["BsmtFullBath"] + 0.5 * (X["BsmtHalfBath"] + X["HalfBath"]) # Total baths
# Categorise categorial variables

# YrSold is also categorical to provide flexibility (esp. due to 2008 financial crisis)

cols = ["MSSubClass", "YrSold"]

X[cols] = X[cols].astype("category")
# Reprsent months as x,y coordinates on a circle to capture the seasonality better

# http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

if 'MoSold' in X:

    X["SinMoSold"] = np.sin(2 * np.pi * X["MoSold"] / 12)

    X["CosMoSold"] = np.cos(2 * np.pi * X["MoSold"] / 12)

    X = X.drop("MoSold", axis=1)
# del X['BsmtFinType1'] # Much worse than new features

# del X['LandSlope'] # Much worse than new feature
# import matplotlib.pyplot as plt



# plt.matshow(X_train.corr())

# plt.show()
# # Add logs (areas, baths, cars, fireplaces, years)

# cols = [

#     'LotFrontage','LotArea','MasVnrArea','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',

#      'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea',

#      'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr',

#      'TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF',

#      'EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YearRemodAdd','TotalSF'

# ]



# for col in cols:

#     X[col+'_log'] = np.log1p(X[col])

# # Add squares

# cols = [

#     'YearRemodAdd', 'LotFrontage_log', 

#     'TotalBsmtSF_log', '1stFlrSF_log', '2ndFlrSF_log', 'GrLivArea_log',

#     'GarageCars_log', 'GarageArea_log',

#     'OverallQual','ExterQual_int','BsmtQual_int','GarageQual_int','FireplaceQu_int','KitchenQual_int'

# ]



# for col in cols:

#     X[col+'_sq'] = np.square(X[col])
# Numerical cols

cols = list(X_train.select_dtypes('number').columns)

fig, axs = plt.subplots(int(np.sqrt(len(cols))), int(np.sqrt(len(cols))), figsize=(30, 30))

for col, ax in zip(cols, axs.ravel()):

    sns.regplot(x=X_train[col], y=y_train, ax=ax);
# Making sure no NaNs left after FE

temp = X.isna().sum()

assert temp.sum() == 0, temp[temp > 0]
# Drop original cols that got ordinalised

print(X.shape)

print(f"Dropping {len(ordinalised_cols)} ordinalised cols: {ordinalised_cols}")

X = X.drop(ordinalised_cols, axis=1)

print(X.shape)
import seaborn as sns

import matplotlib.pyplot as plt



corr=X.corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize = (20,15))

sns.heatmap(corr, mask=mask,  center=0, linewidths=.01, annot=False, cmap='PiYG')
sns.pairplot(pd.concat([X.loc[p_train.index], y_train], axis=1), y_vars='SalePrice', x_vars=sorted(list(X.columns.values)), markers=".", height=3)
skew = X.drop([], axis=1).skew(numeric_only=True).abs()

cols = skew[skew > .5].index

fig, axs = plt.subplots(int(np.sqrt(len(cols))), int(np.sqrt(len(cols))), figsize=(30, 30))

for col, ax in zip(cols, axs.ravel()):

    sns.distplot(X[col], ax=ax)

# X[cols[0]].distplot()
# Transform highly skewed features using boxcox1p and boxcox_normmax, and scale features using RobustScaler.

# Only transform real numeric (not ordinal converted)

# skew = X.drop([col for col in X.columns if '_int' in col], axis=1).skew(numeric_only=True).abs()

# skew = X.drop([col for col in X.columns if ('_int' in col) or ('_sq' in col) or ('_log' in col)], axis=1).skew(numeric_only=True).abs()

# skew = X.drop([col for col in X.columns if ('_sq' in col) or ('_log' in col)], axis=1).skew(numeric_only=True).abs()

skew = X.drop([], axis=1).skew(numeric_only=True).abs()

cols = skew[skew > .5].index

print(skew[skew > .5].sort_values(ascending=False))

for col in cols:

    if X[col].min() > 0: # Error is thrown when negative values exist

        X[col] = boxcox1p(X[col], boxcox_normmax(X[col] + 1))
fig, axs = plt.subplots(int(np.sqrt(len(cols))), int(np.sqrt(len(cols))), figsize=(30, 30))

for col, ax in zip(cols, axs.ravel()):

    sns.distplot(X[col], ax=ax)
cols = X.select_dtypes(np.number).columns

X[cols] = RobustScaler().fit_transform(X[cols])
fig, axs = plt.subplots(int(np.sqrt(len(cols))), int(np.sqrt(len(cols))), figsize=(30, 30))

for col, ax in zip(cols, axs.ravel()):

    sns.distplot(X[col], ax=ax)
# Convert all categorical variables into dummy variables.

X = pd.get_dummies(X)
# temp = X[X.select_dtypes('object').columns].nunique() # TODO High cardinalities to binary

# temp[temp > 5]

# for k, v in temp[temp > 5].iteritems():

#     print(k, ':', X[k].unique())
# Recover train/test features

X_train = X.loc[p_train.index]

X_test = X.loc[p_test.index]
# To remove outliers, we fit a linear model to the training data and remove examples with a studentized residual greater than 3.

residuals = y_train - LinearRegression().fit(X_train, y_train).predict(X_train)

outliers = residuals[np.abs(zscore(residuals)) > 3].index



print(f'Removed {len(outliers)} outliers')

X_train = X_train.drop(outliers)

y_train = y_train.drop(outliers)
# Set up CV strategy (5-folds, RMSE)

kf = KFold(n_splits=5, random_state=0, shuffle=True)

rmse = lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred))

scorer = make_scorer(rmse, greater_is_better=False)
gbm = XGBRegressor(n_estimators=2000, max_depth=4, learning_rate=0.01)
gbm = GradientBoostingRegressor(n_estimators=2000, max_depth=4, learning_rate=0.01)
gbm.fit(X_train, y_train)
[c for c in X_train.columns]
# Get feature importances

feature_importances = gbm.feature_importances_

feature_importances = 100.0 * (feature_importances / feature_importances.max())

fi = pd.DataFrame({'col': X_train.columns, 'val': feature_importances})



# Combine categorical's into one

fi['col_og'], fi['col_suffix'] = fi['col'].str.split('_', 1).str

fi['col_agg'] = np.where((fi['col_suffix']!='int') & (fi['col_suffix']!='sq') & (fi['col_suffix']!='log'),fi['col_og'], fi['col'])

fi_agg = fi.groupby('col_agg').sum().reset_index().sort_values('val', ascending=False).reset_index(drop=True)



# Plot log1p for easier viewing

fi_agg['log_val'] = np.log1p(fi_agg['val'])

fi_agg.plot(kind='barh', figsize=(20,30), x='col_agg', y='log_val')
# Define hyperparam optimisation using random search

def random_search(model, grid, n_iter=100):

    n_jobs = max(cpu_count() - 2, 1)

    search = RandomizedSearchCV(model, grid, n_iter, scorer, n_jobs=n_jobs, cv=kf, random_state=0, verbose=True)

    return search.fit(X_train, y_train)
# Optimise various models (Ridge, Lasso, SVR, LGBM, GBM)

print('Ridge')

ridge_search = random_search(Ridge(), {"alpha": np.logspace(-1, 2, 500)})

print('Lasso')

lasso_search = random_search(Lasso(), {"alpha": np.logspace(-5, -1, 500)})

# print('GAM')

# gam = LinearGAM().gridsearch()

# print('Support Vector Machines')

# svr_search = random_search(SVR(), {"C": np.arange(1, 100), "gamma": np.linspace(0.00001, 0.001, 50), "epsilon": np.linspace(0.01, 0.1, 50)})

# print('LGBM')

# lgbm_search = random_search(LGBMRegressor(n_estimators=4000, max_depth=4), {"colsample_bytree": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})

# print('GBM')

# gbm_search = random_search(GradientBoostingRegressor(n_estimators=4000, max_depth=4), {"max_features": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})

# print('XGB')

# xgb_search = random_search(XGBRegressor(n_estimators=4000, max_depth=4), {"max_features": np.linspace(0.2, 0.7, 6), "learning_rate": np.logspace(-3, -1, 100)})
grid_searches = [

    ridge_search,

    lasso_search,

#     svr_search,

#     lgbm_search,

#     gbm_search,

#     xgb_search,

]



# Get the best models

models = [search.best_estimator_ for search in grid_searches]
# Optimise stacked ensemble of the best models

stack_search = random_search(StackingCVRegressor(models, Ridge(), cv=kf), {"meta_regressor__alpha": np.logspace(-3, -2, 500)}, n_iter=20)

models.append(stack_search.best_estimator_)
preds = [model.predict(X_test) for model in models]
# Average all models (10% weight each) + ensemble (50% weight)

# preds = np.average(preds, axis=0, weights=[0.05] * 4 + [0.8] * 1)

preds = np.average(preds, axis=0, weights=[0.2] * 2 + [0.6] * 1)

# preds = np.average(preds, axis=0, weights=[0.1] * 5 + [0.5] * 1)

# preds = np.average(preds, axis=0, weights=[0.1] * 4 + [0.6] * 1)

# preds = np.average(preds, axis=0, weights=[0.1] * 2 + [0.2] * 2 + [0.4] * 1)
sns.distplot(np.log1p(p_train['SalePrice']))
# # Create submission

submission = pd.DataFrame({"Id": p_sample["Id"], "SalePrice": np.exp(preds)})



# # Copied from other notebooks sort out low and high value predictions

# q1 = submission['SalePrice'].quantile(0.0042)

# q2 = submission['SalePrice'].quantile(0.99)

# # Quantiles helping us get some extreme values for extremely low or high values 

# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)



submission.to_csv("submission.csv", index=False)
# Print out best hyperparams for each model

for search in grid_searches:

    print(search.best_estimator_)

    print(search.best_params_)

    print(search.best_score_)

    print('-'*20)