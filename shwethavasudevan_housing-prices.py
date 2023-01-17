# Basic packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



# Statistics packages

from scipy.stats import norm, skew

from scipy.special import boxcox1p

import statsmodels.api as sm



# sklearn packages for model creation

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

import xgboost as xgb

import lightgbm as lgb
# Import train and test data from .csv



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train.shape
ids = test['Id']    # Submissions file requires an Id column



# Drop Id column from train and test set

train.drop(['Id'], inplace = True, axis = 1)

test.drop(['Id'], inplace = True, axis = 1)



train.head(10)
# Check for missing values in train data



missing_data = train.isna().sum()

missing_data = missing_data.drop(missing_data[missing_data == 0].index)

missing_data
# Check for missing values in test data



missing_test = test.isna().sum()

missing_test = missing_test.drop(missing_test[missing_test == 0].index)

missing_test
corr_matrix = train.corr()

f, ax = plt.subplots(figsize = (12,9))

sns.heatmap(corr_matrix, square = True)
sales_corr = corr_matrix['SalePrice'].sort_values(ascending = False)

sales_corr.drop('SalePrice', axis = 0, inplace = True)

sales_corr.head(10)
train.plot.scatter(x = 'OverallQual', y = 'SalePrice', alpha = 0.4, color = '#00cec9')
train.plot.scatter(x = 'GarageCars', y = 'SalePrice', alpha = 0.3, color = '#f08700')
train.plot.scatter(x = 'GarageArea', y = 'SalePrice', alpha = 0.4, color = '#00cec9')
train.plot.scatter(x = 'TotalBsmtSF', y = 'SalePrice', alpha = 0.3, color = '#f08700')
train.plot.scatter(x = '1stFlrSF', y = 'SalePrice', alpha = 0.4, color = '#00cec9')
# Removal of outliers as stated in the Potential Pitfalls section, refer http://jse.amstat.org/v19n3/decock.pdf

# Houses with GRLIVAREA > 4000 sq ft will be removed - 4 outliers



sns.set_style('darkgrid')

ax = sns.scatterplot(x = train['GrLivArea'], y = train['SalePrice'])



train = train[train['GrLivArea'] < 4000]
# MSZoning - Replacing NaNs with the commonly occuring value (RL)



sns.set_style('white')

plt.subplots(figsize = (6, 5))

sns.set_color_codes('muted')

ax = sns.barplot(x = train['MSZoning'].unique(), y = train.MSZoning.value_counts(), color = 'm', alpha = 0.8)

ax.set(xlabel = 'MSZoning Categories', ylabel = 'Count')



test.MSZoning = test.MSZoning.fillna('RL')
# LotFrontage 

# Houses in the same neighbourhood would probably have similar frontage areas; replace with median



train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

test['LotFrontage'] = test.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
# Alley - Replacing NaNs with 'None'



train['Alley'] = train['Alley'].fillna('None')

test["Alley"] = test['Alley'].fillna('None')

print(train.Alley.isna().sum())

print(test.Alley.isna().sum())
# Utilities 

# Dropping column since only 1 record in train set contains 'NoSeWa' with rest as 'AllPub'



print(train['Utilities'].value_counts())

train.drop(columns = ['Utilities'], inplace = True)

test.drop(columns = ['Utilities'], inplace = True)
# Exterior1st - Replacing with most common occurence



print(train.Exterior1st.mode())

test['Exterior1st'] = test.Exterior1st.fillna('VinylSd')

print(test.Exterior1st.isna().sum())
# Exterior2nd - Replacing with most common occurence



print(train.Exterior2nd.mode())

test['Exterior2nd'] = test.Exterior2nd.fillna('VinylSd')

print(test.Exterior2nd.isna().sum())
# MasVnrType - Replacing NaNs with 'None'



train['MasVnrType'] = train.MasVnrType.fillna('None')

test['MasVnrType'] = test.MasVnrType.fillna('None')
# MasVnrArea

# NaN values in this column all have MasVnrType as None - Hence replacing NaN with 0.0



train.MasVnrArea = train.MasVnrArea.fillna(0.0)

test.MasVnrArea = test.MasVnrArea.fillna(0.0)
# BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2



columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']



for col in columns:

    train[col] = train[col].fillna("None")

    test[col] = test[col].fillna("None")
# BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath, BsmtHalfBath



columns = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']



for col in columns:

    test[col] = test[col].fillna(0)
# Electrical - Replace Nan With the commonly occuring value



print(train.Electrical.mode())

train.Electrical = train.Electrical.fillna('SBrkr')
# KitchenQual - Replace Nan with commonly occuring value



print(train.KitchenQual.mode())

test.KitchenQual = test.KitchenQual.fillna('TA')
# Functional - Assuming typical 'Typ'



test.Functional = test.Functional.fillna('Typ')
# FireplaceQu - NaN means no Fireplace "None"



train['FireplaceQu'] = train['FireplaceQu'].fillna('None')

test['FireplaceQu'] = test['FireplaceQu'].fillna('None')
# GarageType - NaN means no garage "None"



train['GarageType'] = train['GarageType'].fillna('None')

test['GarageType'] = test['GarageType'].fillna('None')
# GarageYrBlt - NaN means no garage, keep year as 0



train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)

test['GarageYrBlt'] = test['GarageYrBlt'].fillna(0)
# GarageFinish - NaN means no garage "None"



train['GarageFinish'] = train['GarageFinish'].fillna("None")

test['GarageFinish'] = test['GarageFinish'].fillna("None")
# GarageCars, GarageArea - NaN means no garage, replace with 0.0



test = test.fillna({'GarageCars':0.0, "GarageArea": 0.0})
# GarageQual, GarageCond - NaN means no garage "None"



for col in ['GarageQual','GarageCond']:

    train[col] = train[col].fillna("None")

    test[col] = test[col].fillna("None")
# PoolQC, Fence - NaN means no Pool/Fence, "None"



for col in ['PoolQC', 'Fence']:

    train[col] = train[col].fillna("None")

    test[col] = test[col].fillna("None")
# MiscFeature



train['MiscFeature'] = train['MiscFeature'].fillna('None')

test['MiscFeature'] = test['MiscFeature'].fillna('None')
# SaleType



print(train['SaleType'].mode())

test['SaleType'].fillna('WD', inplace = True)
# ExterQual, ExterCond, HeatingQC, KitchenQual



exter_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}

for col in ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']:

    train[col] = train[col].map(exter_map).astype(int)

    test[col] = test[col].map(exter_map).astype(int)
# BsmtQual, BsmtCond, FireplaceQu, PoolQC



bsmt_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0}



for col in ['BsmtQual', 'BsmtCond', 'FireplaceQu', 'GarageCond', 'GarageQual', 'PoolQC']:

    train[col] = train[col].map(bsmt_map).astype(int)

    test[col] = test[col].map(bsmt_map).astype(int)
# BsmtExposure



exp_map = {'Gd': 3, 'Av': 2, 'Mn': 1, 'No': 0, 'None': 0}

train['BsmtExposure'] = train['BsmtExposure'].map(exp_map).astype(int)

test['BsmtExposure'] = test['BsmtExposure'].map(exp_map).astype(int)
# BsmtFinType1, BsmtFinType2



fintype_map = {'GLQ': 6, "ALQ": 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}

for col in ['BsmtFinType1', 'BsmtFinType2']:

    train[col] = train[col].map(fintype_map).astype(int)

    test[col] = test[col].map(fintype_map).astype(int)
# CentralAir (Necessary...?)



ca_map = {'Y': 1, 'N': 0}

train['CentralAir'] = train.CentralAir.map(ca_map)

test['CentralAir'] = test.CentralAir.map(ca_map)
# PavedDrive



pd_map = {'Y': 2, 'P': 1, 'N': 0}

train['PavedDrive'] = train.PavedDrive.map(pd_map)

test['PavedDrive'] = test.PavedDrive.map(pd_map)
# Fence



fence_map = {'GdPrv': 2, 'MnPrv': 1, 'GdWo': 2, 'MnWw': 1, 'None': 0}

train['Fence'] = train.Fence.map(fence_map)

test['Fence'] = test.Fence.map(fence_map)
# Street, Alley



street_map = {'Grvl': 2, 'Pave': 1, 'None': 0}

for col in ['Street', 'Alley']:

    train[col] = train[col].map(street_map)

    test[col] = test[col].map(street_map)
# LotShape



lot_map = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}

train['LotShape'] = train.LotShape.map(lot_map)

test['LotShape'] = test.LotShape.map(lot_map)
# LandSlope



slope_map = {'Gtl': 3, 'Mod': 2, 'Sev': 1}

train['LandSlope'] = train.LandSlope.map(slope_map)

test['LandSlope'] = test.LandSlope.map(slope_map)
# MasVnrType



mason_map = {'BrkCmn': 1, 'BrkFace': 1, 'CBlock': 1, 'Stone': 1, 'None': 0}

train['MasVnrType'] = train.MasVnrType.map(mason_map)

test['MasVnrType'] = test.MasVnrType.map(mason_map)
# OverallQC - from OverallQual and OverallCond



train['OverallQC'] = (train['OverallQual'] + train['OverallCond']) / 2

test['OverallQC'] = (test['OverallQual'] + test['OverallCond']) / 2



train.drop(['OverallQual', 'OverallCond'], inplace = True, axis = 1)

test.drop(['OverallQual', 'OverallCond'], inplace = True, axis = 1)
# ExterQC - from ExterQual and ExterCond



train['ExterQC'] = (train['ExterQual'] + train['ExterCond']) / 2

test['ExterQC'] = (test['ExterQual'] + test['ExterCond']) / 2



train.drop(['ExterQual', 'ExterCond'], inplace = True, axis = 1)

test.drop(['ExterQual', 'ExterCond'], inplace = True, axis = 1)
# GarageQC - from GarageQual and GarageCond



train['GarageQC'] = (train['GarageQual'] + train['GarageCond']) / 2

test['GarageQC'] = (test['GarageQual'] + test['GarageCond']) / 2



train.drop(['GarageQual', 'GarageCond'], inplace = True, axis = 1)

test.drop(['GarageQual', 'GarageCond'], inplace = True, axis = 1)
# BsmtQC - from BsmtQual, BsmtCond, BsmtExposure



train['BsmtQC'] = (train['BsmtQual'] + train['BsmtCond'] + train['BsmtExposure']) / 3

test['BsmtQC'] = (test['BsmtQual'] + test['BsmtCond'] + test['BsmtExposure']) / 3



train.drop(['BsmtQual', 'BsmtCond', 'BsmtExposure'], axis = 1, inplace = True)

test.drop(['BsmtQual', 'BsmtCond', 'BsmtExposure'], axis = 1, inplace = True)
# TopFloorsArea - from 1stFlrSF, 2ndFlrSF

# TotalSFArea - from GrLivArea, TotalBsmtSF



train['TopFloorsArea'] = train['1stFlrSF'] + train['2ndFlrSF']  

train['TotalSFArea'] = train['GrLivArea'] + train['TotalBsmtSF']

test['TopFloorsArea'] = test['1stFlrSF'] + test['2ndFlrSF']  

test['TotalSFArea'] = test['GrLivArea'] + test['TotalBsmtSF']



train.drop(['1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotalBsmtSF'], inplace = True, axis = 1)

test.drop(['1stFlrSF', '2ndFlrSF', 'GrLivArea', 'TotalBsmtSF'], inplace = True, axis = 1)
# TotBaths - Total number of bathrooms from BsmtFullBath, BsmtHalfBath, HalfBath, FullBath.



train['TotBaths'] = train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath']) + (0.5 * train['HalfBath']) + train['FullBath']

test['TotBaths'] = test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath']) + (0.5 * test['HalfBath']) + test['FullBath']



train.drop(['BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'FullBath'], inplace = True, axis = 1)

test.drop(['BsmtFullBath', 'BsmtHalfBath', 'HalfBath', 'FullBath'], inplace = True, axis = 1)
# TotPorchSF - Total Square feet area of the porch.



train['TotPorchSF'] = train['OpenPorchSF'] + train['EnclosedPorch'] + train['3SsnPorch'] + train['ScreenPorch']

test['TotPorchSF'] = test['OpenPorchSF'] + test['EnclosedPorch'] + test['3SsnPorch'] + test['ScreenPorch']



train.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], inplace = True, axis = 1)

test.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch'], inplace = True, axis = 1)
# Dropping the GarageCars, GarageYrBuilt columns.



train.drop(['GarageCars', 'GarageYrBlt'], inplace = True, axis = 1)

test.drop(['GarageCars', 'GarageYrBlt'], inplace = True, axis = 1)
# Converting the categorical features to Pandas Categorical type.

# This is done to distinguish them from the numerical features that are to be transformed.



columns = ['MSSubClass', 'MSZoning', 'LandContour', 'LotConfig', 

           'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

           'HouseStyle', 'YearBuilt', 'YearRemodAdd', 'RoofStyle',

           'RoofMatl', 'Exterior1st', 'Exterior2nd', 'Foundation', 

           'Heating', 'Electrical', 'Functional', 'GarageType', 

           'GarageFinish', 'PavedDrive', 'MiscFeature', 'MoSold', 

           'YrSold', 'SaleType', 'SaleCondition']



num_train = train.shape[0]

num_test = test.shape[0]



y_train = train['SalePrice'].values    # SalePrice 

train.drop(['SalePrice'], axis = 1, inplace = True)



all_data = pd.concat([train, test], axis = 0).reset_index(drop = True)



for col in columns:

    all_data[col] = pd.Categorical(all_data[col])
# Obtaining the column containing numerical features



num_features = all_data.dtypes[all_data.dtypes != 'category'].index

print(num_features)
# Skewness with magnitude 0.5 is considered to be moderate.

# A skewness magnitude greater than 1 is considered as high.



skewness = all_data[num_features].apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

skewed_features = skewness.index
all_data[skewed_features] = np.log1p(all_data[skewed_features])    
# Using the get_dummies() method from pandas to obtain dummy variables.



for col in columns:

    all_data = pd.concat([all_data, pd.get_dummies(all_data[col], prefix = col, drop_first = True)], axis = 1)



# Drop original columns



all_data.drop(columns, inplace = True, axis = 1)

print(all_data.shape)

all_data.head(5)
X_train = all_data[:num_train]

X_test = all_data[num_train:]
scaler = StandardScaler()

X_train.loc[:, num_features] = scaler.fit_transform(X_train.loc[:, num_features])

X_test.loc[:, num_features] = scaler.transform(X_test.loc[:, num_features])
ax = sns.distplot(y_train, fit = norm)

ax.set_title('SalePrice distribution')

print("Skewness: ", skew(y_train))
sm.qqplot(y_train, line = 's')

plt.title('QQ Plot')

plt.show()
y_train = np.log1p(y_train)
# Plotting the distribution and QQ plot



ax = sns.distplot(y_train, fit = norm)

ax.set_title('SalePrice distribution')



sm.qqplot(y_train, line = 's')

plt.title('After log1p transformation')

plt.show()
rmse_scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv(model):

    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = rmse_scorer, cv = 10))

    return (rmse)
ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1.0, 3, 6, 10, 30, 60])

ridge.fit(X_train, y_train)



alpha = ridge.alpha_

print("Estimated alpha value: ",alpha)



# Phase 2 - Ridge



ridge = RidgeCV(alphas = [alpha * 0.5, alpha * 0.6, alpha * 0.7, alpha * 0.8, alpha * 0.9, alpha * 1,

                         alpha * 1.1, alpha * 1.2, alpha * 1.3, alpha * 1.4, alpha * 1.5])

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Estimated alpha value (Phase 2):", alpha)

print()

score = rmse_cv(ridge)

print("Mean Ridge Score:", score.mean())

print("Std. Deviation:", score.std())
lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

                          0.01, 0.06, 0.1, 0.3, 0.6, 1], random_state = 0, cv = 10, max_iter = 50000)

lasso.fit(X_train, y_train)



alpha = lasso.alpha_

print("Estimated alpha value: ", alpha)



# Phase 2 - Lasso



lasso = LassoCV(alphas = [alpha * 0.5, alpha * 0.6, alpha * 0.7, alpha * 0.8, alpha * 0.9, alpha * 1,

                         alpha * 1.1, alpha * 1.2, alpha * 1.3, alpha * 1.4, alpha * 1.5], random_state = 0, 

                         max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Estimated alpha value (Phase 2):", alpha)

print()

score = rmse_cv(lasso)

print("Mean Lasso Score:", score.mean())

print("Std. Deviation:", score.std())
enet = ElasticNetCV(l1_ratio = [0.1, 0.15, 0.3, 0.55, 0.7, 0.95, 1],

                         alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 

                                   0.006, 0.1, 0.3, 0.6, 0.9, 1, 3, 6, 12],

                   max_iter = 50000, cv = 10)

enet.fit(X_train, y_train)

ratio = enet.l1_ratio_

alpha = enet.alpha_

print("Best l1 ratio:", ratio)

print("Best alpha:", alpha)



# Phase 2 - ElasticNet



enet = ElasticNetCV(l1_ratio = ratio,

                   alphas = [alpha * 0.5, alpha * 0.6, alpha * 0.7, alpha * 0.8, alpha * 0.9, alpha * 1,

                         alpha * 1.1, alpha * 1.2, alpha * 1.3, alpha * 1.4, alpha * 1.5], random_state = 0, 

                         max_iter = 50000, cv = 10)

enet.fit(X_train, y_train)

ratio = enet.l1_ratio_

alpha = enet.alpha_

print("Best l1 ratio (Phase 2):", ratio)

print("Best alpha (Phase 2):", alpha)

print()

score = rmse_cv(enet)

print("Mean ElasticNet score:", score.mean())

print("Std. Deviation:", score.std())
y_pred = np.expm1(enet.predict(X_test))

y_pred 
# Creating the submissions .csv file



submissions = pd.DataFrame({

    

    'Id': ids,

    'SalePrice': y_pred

})



submissions.to_csv('submissions.csv', index = False)