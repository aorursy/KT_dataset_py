# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, BayesianRidge, HuberRegressor, GammaRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.feature_selection import RFE
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
# Read train and test data
train_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test_raw = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
# Check the shape of train and test data
print(train_raw.shape)
print(test_raw.shape)
# Inspect train data using head() to get a general idea
# We can see that there are some columns with NaN values which we will deal later
train_raw.head()
# Check datatypes for all columns/features
# We'll only check for train data as test data has same features except the target feature SalePrice
# We can observe that there are int, float and string values present in the dataset
# We'll seperate numeric and string features, to be used later
numeric_cols = []
string_cols = []
for col in train_raw.columns:
    print(col," : ", train_raw[col].dtype)
    if (train_raw[col].dtype in ["int","float"]):
        numeric_cols.append(col)
    elif (train_raw[col].dtype in ["object"]):
        string_cols.append(col)

# Check numeric and string features
print("\nNumeric Features: ",len(numeric_cols), "\n", numeric_cols, "\n\n")
print("\nString Features: ", len(string_cols), "\n",string_cols)
# We'll perform the same data preprocessing for both train and test data but seperately to avoid data leak

# Check for missing values in numeric features for train data
numeric_missing_cols_train = []
for col in numeric_cols:
    if(train_raw[col].isnull().sum() > 0):
        print(col, " : ", train_raw[col].isnull().sum())
        numeric_missing_cols_train.append(col)

# Numeric columns with missing values for train data
print("Numeric columns train data: ",numeric_missing_cols_train,"\n")

# Check for missing values in numeric features(except target feature SalePrice) for test data
numeric_missing_cols_test = []
for col in [numeric_col for numeric_col in numeric_cols if numeric_col != "SalePrice"]:
    if(test_raw[col].isnull().sum() > 0):
        print(col, " : ", test_raw[col].isnull().sum())
        numeric_missing_cols_test.append(col)

# Numeric columns with missing values for test data
print("Numeric columns test data: ",numeric_missing_cols_test)
# Check for zero values in numeric features
# We'll deal with these zero values later on
# Not all zero values are useless
print("Train features with zero values: \n",train_raw[numeric_cols].isin([0]).sum(),"\n\n")
print("Test features with zero values: \n",test_raw[[numeric_col for numeric_col in numeric_cols if numeric_col != "SalePrice"]].isin([0]).sum())
# Explore numeric columns with missing values for train data
print(train_raw[numeric_missing_cols_train])

# Check the relationship of these features with the target variable SalePrice using jointplot
# We can observe some large values which can be considered as outliers provided that we are dealing with regression
for i, feature in enumerate(numeric_missing_cols_train, 1):
    sns.jointplot(x=feature, y='SalePrice', data=train_raw, kind = 'reg', height = 5)
plt.show()
# Check for missing values in string features for train data
string_missing_cols_train = []
for col in string_cols:
    if(train_raw[col].isnull().sum() > 0):
        print(col, " : ", train_raw[col].isnull().sum())
        string_missing_cols_train.append(col)
        
# String columns with missing data for train data
print("String columns test data: ",string_missing_cols_train, "\n")

# Check for missing values in string features for test data
string_missing_cols_test = []
for col in string_cols:
    if(test_raw[col].isnull().sum() > 0):
        print(col, " : ", test_raw[col].isnull().sum())
        string_missing_cols_test.append(col)
        
# String columns with missing data for test data
print("String columns test data: ",string_missing_cols_test)
# Find correlation among the numeric features of the train data
# Corr() ignores NaN and non-numeric values
cor = train_raw.corr()
corr_target = abs(cor['SalePrice'])

relevant_feature_names = []
for col, value in corr_target[corr_target>0.3].iteritems():
    relevant_feature_names.append(col)

relevant_feature_names
# Copy train_raw and test_raw for preprocessing
train = train_raw.copy()
test = test_raw.copy()
# Handling numeric missing values for train data
# LotFrontage  :  259
# MasVnrArea  :  8
# GarageYrBlt  :  81

# Inspect Garage features to see if GarageYrBlt values are missing 
# because of No Garage value in their corresponding Garage features

# Get all row indexes that have missing values for GarageYrBlt feature 
garage_row_list = train.index[train['GarageYrBlt'].isnull()].tolist()
# Get col indexes for Garage features
garage_col_list = [train.columns.get_loc(col) for col in ['GarageType', 'GarageYrBlt', 'GarageFinish','GarageArea', 'GarageQual']]
print("Garage feature rows: ", garage_row_list)
print("Garage feature cols: ", garage_col_list)
# We can observe below that GarageYrBlt feature has missing values
# because its corresponding Garage features have No Garage(NaN) as their values 
print(train.iloc[garage_row_list, garage_col_list])
# Impute 0 for missing values of numerical feature GarageYrBlt as the corresponding GarageType value is No Garage (NaN)
train['GarageYrBlt'] = train['GarageYrBlt'].fillna(0)
# Check min, max, mean, mode values for numeric features with missing values
print("Minimum: ",train['LotFrontage'].min(), train['MasVnrArea'].min(), train['GarageYrBlt'].min())
print("Average: ",train['LotFrontage'].mean(), train['MasVnrArea'].mean(), train['GarageYrBlt'].mean())
print("Mode: ",train['LotFrontage'].mode()[0], train['MasVnrArea'].mode()[0], train['GarageYrBlt'].mode()[0])
print("Maximum: ",train['LotFrontage'].max(), train['MasVnrArea'].max(), train['GarageYrBlt'].max())

# Use mean value to impute the mising values in two numeric columns: LotFrontage, MasVnrArea
# We'll handle GarageYrBlt later
for col in [nmc_train for nmc_train in numeric_missing_cols_train if nmc_train != "GarageYrBlt"]:
    train[col] = train[col].fillna(train[col].mean())
# Handling numeric missing values for test data
# LotFrontage  :  227
# MasVnrArea  :  15
# BsmtFinSF1  :  1
# BsmtFinSF2  :  1
# BsmtUnfSF  :  1
# TotalBsmtSF  :  1
# BsmtFullBath  :  2
# BsmtHalfBath  :  2
# GarageYrBlt  :  78
# GarageCars  :  1
# GarageArea  :  1

# Inspect Garage features to see if GarageYrBlt values are missing 
# because of No Garage value in their corresponding Garage features

# Get all row indexes that have missing values for GarageYrBlt feature 
garage_row_list = test.index[test['GarageYrBlt'].isnull()].tolist()
garage_row_list = garage_row_list + test.index[test['GarageCars'].isnull()].tolist()
garage_row_list = garage_row_list + test.index[test['GarageArea'].isnull()].tolist()
# Get col indexes for Garage features
garage_col_list = [test.columns.get_loc(col) for col in 
                   ['GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 'GarageCars', 'GarageArea']]
print("Garage feature rows: ", garage_row_list)
print("Garage feature cols: ", garage_col_list)
# We can observe below that GarageYrBlt feature has missing values
# because its corresponding Garage features have No Garage(NaN) as their values 
# The GarageCars, GarageArea are missing but 
# their corresponding GarageFinish, GarageQual, GarageCond features have No Garage(NaN) values
print(test.iloc[garage_row_list, garage_col_list])

# Corresponding No basement values can be observed below
bsmt_row_list = test.index[test['BsmtFinSF1'].isnull()].tolist() + \
                test.index[test['BsmtFullBath'].isnull()].tolist()
print("BsmtFinSF1", bsmt_row_list) 
print(test.iloc[bsmt_row_list, [31,30,34,36,37,38,47,48]])
# Use mean value to impute the mising values in two numeric columns: LotFrontage, MasVnrArea
for col in ['LotFrontage', 'MasVnrArea']:
    test[col] = test[col].fillna(test[col].mean())
# Impute 0 for missing values of numerical feature GarageYrBlt as the corresponding GarageType value is No Garage(NaN)
for col in ['GarageYrBlt', 'GarageCars', 'GarageArea']:
    test[col] = test[col].fillna(0)
# Impute 0 for missing values of numerical Bsmt features as the corresponding Bsmt values are No Basement(NaN)
for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
    test[col] = test[col].fillna(0)
# As per the given description of the dataset, not all columns 
# with NaN values are considered as missing values
# NaN is used as an option in categorical data for most of the string features
# Observe unique values for all string features to check how many are categorical
# Observation suggest all columns are categorical so we can apply an encoding scheme
# And consider NaN values in the categorical features as an option which is 
# mentioned in the data description
print("Feature Name", "Unique Values")
for col in string_cols:
    print(col, train[col].unique())
# Handling string missing values for train data
# Alley  :  1369 "NA-No alley access"
# MasVnrType  :  8 "None-None"
# BsmtQual  :  37 "NA-No basement"
# BsmtCond  :  37 "NA-No basement"
# BsmtExposure  :  38 "NA-No basement"
# BsmtFinType1  :  37 "NA-No basement"
# BsmtFinType2  :  38 "NA-No basement"
# Electrical  :  1
# FireplaceQu  :  690, "NA-No Fireplace"
# GarageType  :  81 "NA-No Garage"
# GarageFinish  :  81 "NA-No Garage"
# GarageQual  :  81 "NA-No Garage"
# GarageCond  :  81 "NA-No Garage"
# PoolQC  :  1453 "NA-No Pool"
# Fence  :  1179 "NA-No Fence"
# MiscFeature  :  1406 "NA-None"

# Replace all NaN values in the string features with None except Electrical
for col in [strmc for strmc in string_missing_cols_train if strmc != "Electrical"]:
    train[col] = train[col].fillna("None")
# Since Electrical has only 1 missing value, we can use mode to replace it which is SBrkr
train['Electrical'] = train["Electrical"].fillna(train["Electrical"].mode()[0])
# Intermediate copy
train1 = train.copy()
test1 = test.copy()
# Handling string missing values for train data
# MSZoning  :  4
# Alley  :  1352 "NA-No alley access"
# Utilities  :  2
# Exterior1st  :  1
# Exterior2nd  :  1
# MasVnrType  :  16 "None-None"
# BsmtQual  :  44 "NA-No basement"
# BsmtCond  :  45 "NA-No basement"
# BsmtExposure  :  44 "NA-No basement"
# BsmtFinType1  :  42 "NA-No basement"
# BsmtFinType2  :  42 "NA-No basement"
# KitchenQual  :  1
# Functional  :  2 (Assume typical unless deductions are warranted)
# FireplaceQu  :  730 "NA-No Fireplace"
# GarageType  :  76 "NA-No Garage"
# GarageFinish  :  78 "NA-No Garage"
# GarageQual  :  78 "NA-No Garage"
# GarageCond  :  78 "NA-No Garage"
# PoolQC  :  1456 "NA-No Pool"
# Fence  :  1169 "NA-No Fence"
# MiscFeature  :  1408 "NA-None"
# SaleType  :  1

# Replace all NaN values in the string features with None 
# except MSZoning, Utilities, Exterior1st, Exterior2nd, KitchenQual, Functional, SaleType
for col in [strmc for strmc in string_missing_cols_test if strmc not in 
            ['MSZoning','Utilities','Exterior1st','Exterior2nd','KitchenQual','Functional','SaleType']]:
    test1[col] = test1[col].fillna("None")
# Impute Typ value for missing Functional values based on data description
test1['Functional'] = test1['Functional'].fillna('Typ')
# Impute mode value for remaining categorical features
test1['MSZoning'] = test1['MSZoning'].fillna(test1['MSZoning'].mode()[0])
test1['Utilities'] = test1['Utilities'].fillna(test1['Utilities'].mode()[0])
test1['Exterior1st'] = test1['Exterior1st'].fillna(test1['Exterior1st'].mode()[0])
test1['KitchenQual'] = test1['KitchenQual'].fillna(test1['KitchenQual'].mode()[0])
test1['SaleType'] = test1['SaleType'].fillna(test1['SaleType'].mode()[0])
# Use the 2nd most frequent option: MetalSd 
test1['Exterior2nd'] = test1['Exterior2nd'].fillna('MetalSd')
# Encode categorical features using LabelEncoder
le = LabelEncoder()
for col in string_cols:
    train1[col] = le.fit_transform(train1[col])
    test1[col] = le.fit_transform(test1[col])

print(train1[string_cols].head())
print(test1[string_cols].head())
train1 = train1.astype("int64")
test1 = test1.astype("int64")
# Find correlation after handling categorical values
cor1 = train1.corr()
corr_target1 = abs(cor1['SalePrice'])

relevant_feature_names1 = []
for col, value in corr_target1[corr_target1>0.3].iteritems():
    relevant_feature_names1.append(col)

relevant_feature_names1
# Set _has_statsmodels to False to plot features that have data not well represented using gaussian distribution
# In this dataset, these features have a lot of zero values or a lot of indexes with same values
# These features don't work well with regression
# Data is not continous so it is treated as categorical
# sns.distributions._has_statsmodels = False

# sns.jointplot(x='BsmtFinSF2', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='LowQualFinSF', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='BsmtHalfBath', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='KitchenAbvGr', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='EnclosedPorch', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='3SsnPorch', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='ScreenPorch', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='PoolArea', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='MiscVal', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)

# plt.show()
# Joint plots for numerical features
# sns.jointplot(x='Id', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='MSSubClass', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='LotFrontage', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='LotArea', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='OverallQual', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='OverallCond', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='YearBuilt', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='YearRemodAdd', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='MasVnrArea', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='BsmtFinSF1', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='BsmtUnfSF', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='TotalBsmtSF', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='1stFlrSF', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='2ndFlrSF', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='GrLivArea', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='BsmtFullBath', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='FullBath', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='HalfBath', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='BedroomAbvGr', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='TotRmsAbvGrd', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='Fireplaces', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='GarageYrBlt', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='GarageCars', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='GarageArea', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='WoodDeckSF', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='OpenPorchSF', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='MoSold', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)
# sns.jointplot(x='YrSold', y='SalePrice', data=train1[numeric_cols], kind = 'reg', height = 5)

# plt.show()
# Numeric Features that show linearity with SalePrice
# We can remove outliers from these columns to improve regression result

# ['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1','BsmtUnfSF', 'TotalBsmtSF', 
#  '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 
#  'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
# Submission['SalePrice'] will be used as y_test.
# This is done to calculate the test metrics and get an idea about the results.
# The answers of the metrics do not represent their actual values as the real y_test may be different
submission = pd.read_csv('../input/house-prices-advanced-regression-techniques/sample_submission.csv')
submission.head()
# Check the variance of all numerical features
# Also check the normalized variance
# We can observe that the normalized variance looks better
pd.DataFrame([train1[numeric_cols].var(),np.log1p(train1[numeric_cols]).var()])
# Standard Scalar. If data is not scaled we can get negative score

# Add SalePrice to test data for scaling
# test_temp = test1.copy()
# test_temp['SalePrice'] = submission['SalePrice']

sscaler = StandardScaler()
scaled_train = pd.DataFrame(sscaler.fit(train1.drop(["SalePrice"], axis = 1))
                            .transform(train1.drop(["SalePrice"], axis = 1)),
                            columns=train1.drop(["SalePrice"], axis = 1).columns)
scaled_test = pd.DataFrame(sscaler.transform(test1), columns=test1.columns)
# Minmax Scalar. If data is not scaled we can get negative score

# Add SalePrice to test data for scaling
# test_temp = test1.copy()
# test_temp['SalePrice'] = submission['SalePrice']

mmscaler = MinMaxScaler()
scaled_train = pd.DataFrame(mmscaler.fit(train1.drop(["SalePrice"], axis = 1))
                            .transform(train1.drop(["SalePrice"], axis = 1)),
                            columns=train1.drop(["SalePrice"], axis = 1).columns)
scaled_test = pd.DataFrame(mmscaler.transform(test1), columns=test1.columns)
# Check the variance of all numerical features
# Also check the scaled variance
pd.DataFrame([train1[numeric_cols].var(),(scaled_train[[col for col in numeric_cols if col!="SalePrice"]]).var()])
# Test 8- Normalized data
# X_train = np.log1p(train1[['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1','BsmtUnfSF', 'TotalBsmtSF', 
#  '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 
#  'GarageArea', 'WoodDeckSF', 'OpenPorchSF']])
# y_train = np.log1p(train1["SalePrice"])
# X_test = np.log1p(test1[['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1','BsmtUnfSF', 'TotalBsmtSF', 
#  '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 
#  'GarageArea', 'WoodDeckSF', 'OpenPorchSF']])
# y_test = np.log1p(submission['SalePrice'])
# Test 6- Scale
X_train = scaled_train.drop(["Id"], axis = 1)
y_train = np.log1p(train1["SalePrice"])
X_test = scaled_test.drop(["Id"], axis = 1)
y_test = np.log1p(submission['SalePrice'])
# Test 5- Linear numeric features from joint plot
# X_train = train1[['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1','BsmtUnfSF', 'TotalBsmtSF', 
#  '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 
#  'GarageArea', 'WoodDeckSF', 'OpenPorchSF']]
# y_train = np.log1p(train1["SalePrice"])

# X_test = test1[['LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'MasVnrArea', 'BsmtFinSF1','BsmtUnfSF', 'TotalBsmtSF', 
#  '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 
#  'GarageArea', 'WoodDeckSF', 'OpenPorchSF']]
# y_test = np.log1p(submission['SalePrice'])
# Test 4- Correlation numerical and categorical
# X_train = train1[['LotFrontage','OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','ExterQual','Foundation','BsmtQual','BsmtExposure','BsmtFinSF1','TotalBsmtSF','HeatingQC','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','Fireplaces','GarageType','GarageFinish','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF']].astype('int64')
# y_train = np.log1p(train1["SalePrice"].astype('int64'))
# X_test = test1[['LotFrontage','OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','ExterQual','Foundation','BsmtQual','BsmtExposure','BsmtFinSF1','TotalBsmtSF','HeatingQC','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','Fireplaces','GarageType','GarageFinish','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF']].astype('int64')
# y_test = np.log1p(submission['SalePrice'].astype('int64'))
# Test 3- Feature importance
# X_train = train1[['PoolArea','MasVnrType','BsmtFinType2','LowQualFinSF','BsmtFinSF2','PoolQC']].astype('int64')
# y_train = np.log1p(train1["SalePrice"].astype('int64'))
# X_test = test1[['PoolArea','MasVnrType','BsmtFinType2','LowQualFinSF','BsmtFinSF2','PoolQC']].astype('int64')
# y_test = np.log1p(submission['SalePrice'].astype('int64'))
# Test 2- Correlation only numerical
# X_train = train1[['LotFrontage','OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF']].astype('int64')
# y_train = np.log1p(train1["SalePrice"].astype('int64'))
# X_test = test1[['LotFrontage','OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF','OpenPorchSF']].astype('int64')
# y_test = np.log1p(submission['SalePrice'].astype('int64'))
# Test 1- Only handle missing values
# X_train = train1.drop(["Id","SalePrice"], axis = 1)
# y_train = np.log1p(train1["SalePrice"])
# X_test = test1.drop(["Id"], axis = 1)
# y_test = np.log1p(submission['SalePrice'])
# Model Building
lr = LinearRegression()

# lr_selector = RFE(lr, n_features_to_select=20)
# lr_selector = lr_selector.fit(X_train, y_train)

# lr_train_score = lr_selector.score(X_train, y_train)
# lr_test_score = lr_selector.score(X_test, y_test)
# lr_train_predict = lr_selector.predict(X_train)
# lr_test_predict = lr_selector.predict(X_test)

lr.fit(X_train, y_train)

lr_train_predict = lr.predict(X_train)
lr_test_predict = lr.predict(X_test)
lr_train_score = r2_score(y_train,lr_train_predict)
lr_test_score = r2_score(y_test,lr_test_predict)
# lr_train_score = lr.score(X_train, y_train)
# lr_test_score = lr.score(X_test, y_test)

# Get the selected column names
for i in range(X_train.shape[1]):
    if(lr_selector.support_[i] == True):
        print('Column: %d, %s, Selected %s, Rank: %.3f' % (i,X_train.columns[i], lr_selector.support_[i], lr_selector.ranking_[i]))
rr = Ridge()
rr_selector = RFE(rr, n_features_to_select=20)
rr_selector = rr_selector.fit(X_train, y_train)

rr_train_score = rr_selector.score(X_train, y_train)
rr_test_score = rr_selector.score(X_test, y_test)
rr_train_predict = rr_selector.predict(X_train)
rr_test_predict = rr_selector.predict(X_test)

# rr.fit(X_train, y_train)

# rr_train_score = rr.score(X_train, y_train)
# rr_test_score = rr.score(X_test, y_test)
# rr_train_predict = rr.predict(X_train)
# rr_test_predict = rr.predict(X_test)
# Get the selected column names
for i in range(X_train.shape[1]):
    if(rr_selector.support_[i] == True):
        print('Column: %d, %s, Selected %s, Rank: %.3f' % (i,X_train.columns[i], rr_selector.support_[i], rr_selector.ranking_[i]))
lar = Lasso()
lar_selector = RFE(lar, n_features_to_select=20)
lar_selector = lar_selector.fit(X_train, y_train)

lar_train_score = lar_selector.score(X_train, y_train)
lar_test_score = lar_selector.score(X_test, y_test)
lar_train_predict = lar_selector.predict(X_train)
lar_test_predict = lar_selector.predict(X_test)

# lar.fit(X_train, y_train)

# lar_train_score = lar.score(X_train, y_train)
# lar_test_score = lar.score(X_test, y_test)
# lar_train_predict = lar.predict(X_train)
# lar_test_predict = lar.predict(X_test)
# Get the selected column names
for i in range(X_train.shape[1]):
    if(lar_selector.support_[i] == True):
        print('Column: %d, %s, Selected %s, Rank: %.3f' % (i,X_train.columns[i], lar_selector.support_[i], lar_selector.ranking_[i]))
bar = BayesianRidge(n_iter=1000)
bar_selector = RFE(bar, n_features_to_select=20)
bar_selector = bar_selector.fit(X_train, y_train)

bar_train_score = bar_selector.score(X_train, y_train)
bar_test_score = bar_selector.score(X_test, y_test)
bar_train_predict = bar_selector.predict(X_train)
bar_test_predict = bar_selector.predict(X_test)

# bar.fit(X_train, y_train)

# bar_train_score = bar.score(X_train, y_train)
# bar_test_score = bar.score(X_test, y_test)
# bar_train_predict = bar.predict(X_train)
# bar_test_predict = bar.predict(X_test)
# Get the selected column names
for i in range(X_train.shape[1]):
    if(bar_selector.support_[i] == True):
        print('Column: %d, %s, Selected %s, Rank: %.3f' % (i,X_train.columns[i], bar_selector.support_[i], bar_selector.ranking_[i]))
hr = HuberRegressor(max_iter=3000)
# hr_selector = RFE(hr, n_features_to_select=20)
# hr_selector = hr_selector.fit(X_train, y_train)

# hr_train_score = hr_selector.score(X_train, y_train)
# hr_test_score = hr_selector.score(X_test, y_test)
# hr_train_predict = hr_selector.predict(X_train)
# hr_test_predict = hr_selector.predict(X_test)

hr.fit(X_train, y_train)

hr_train_score = hr.score(X_train, y_train)
hr_test_score = hr.score(X_test, y_test)
hr_train_predict = hr.predict(X_train)
hr_test_predict = hr.predict(X_test)
rfr = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1, random_state=0)
rfr_selector = RFE(rfr, n_features_to_select=15)
rfr_selector = rfr_selector.fit(X_train, y_train)

rfr_train_score = rfr_selector.score(X_train, y_train)
rfr_test_score = rfr_selector.score(X_test, y_test)
rfr_train_predict = rfr_selector.predict(X_train)
rfr_test_predict = rfr_selector.predict(X_test)

# rfr.fit(X_train, y_train)

# rfr_train_score = rfr.score(X_train, y_train)
# rfr_test_score = rfr.score(X_test, y_test)
# rfr_train_predict = rfr.predict(X_train)
# rfr_test_predict = rfr.predict(X_test)
# Get the selected column names
for i in range(X_train.shape[1]):
    if(rfr_selector.support_[i] == True):
        print('Column: %d, %s, Selected %s, Rank: %.3f' % (i,X_train.columns[i], rfr_selector.support_[i], rfr_selector.ranking_[i]))
importances = rfr.feature_importances_
indices = np.argsort(importances)
indices = indices[1:10]
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [X_train.columns[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
abr = AdaBoostRegressor(n_estimators=100, random_state=0)
abr_selector = RFE(abr, n_features_to_select=15)
abr_selector = abr_selector.fit(X_train, y_train)

abr_train_score = abr_selector.score(X_train, y_train)
abr_test_score = abr_selector.score(X_test, y_test)
abr_train_predict = abr_selector.predict(X_train)
abr_test_predict = abr_selector.predict(X_test)

# abr.fit(X_train, y_train)

# abr_train_score = abr.score(X_train, y_train)
# abr_test_score = abr.score(X_test, y_test)
# abr_train_predict = abr.predict(X_train)
# abr_test_predict = abr.predict(X_test)
# Get the selected column names
for i in range(X_train.shape[1]):
    if(abr_selector.support_[i] == True):
        print('Column: %d, %s, Selected %s, Rank: %.3f' % (i,X_train.columns[i], abr_selector.support_[i], abr_selector.ranking_[i]))
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01, max_depth=3, loss='huber', random_state=0)
# gbr_selector = RFE(gbr, n_features_to_select=20)
# gbr_selector = gbr_selector.fit(X_train, y_train)

# gbr_train_score = gbr_selector.score(X_train, y_train)
# gbr_test_score = gbr_selector.score(X_test, y_test)
# gbr_train_predict = gbr_selector.predict(X_train)
# gbr_test_predict = gbr_selector.predict(X_test)

gbr.fit(X_train, y_train)

gbr_train_score = gbr.score(X_train, y_train)
gbr_test_score = gbr.score(X_test, y_test)
gbr_train_predict = gbr.predict(X_train)
gbr_test_predict = gbr.predict(X_test)
xgbr = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.01,
                max_depth = 3, n_estimators = 6000)
# xgbr_selector = RFE(xgbr, n_features_to_select=20)
# xgbr_selector = xgbr_selector.fit(X_train, y_train)

# xgbr_train_score = xgbr_selector.score(X_train, y_train)
# xgbr_test_score = xgbr_selector.score(X_test, y_test)
# xgbr_train_predict = xgbr_selector.predict(X_train)
# xgbr_test_predict = xgbr_selector.predict(X_test)

xgbr.fit(X_train, y_train)

xgbr_train_score = xgbr.score(X_train, y_train)
xgbr_test_score = xgbr.score(X_test, y_test)
xgbr_train_predict = xgbr.predict(X_train)
xgbr_test_predict = xgbr.predict(X_test)
# Takes a bit of time to execute
# estimators = [('lr',lr),('rr',rr),('lar',lar),('bar',bar), ('hr',hr), ('abr',abr), ('gbr',gbr), ('xgbr',xgbr)]
estimators = [('lr',lr),('rr',rr),('lar',lar),('bar',bar), ('abr',abr)]
# estimators = [('lr',lr),('rr',rr),('bar',bar),('rfr',rfr), ('abr',abr),('gbr',gbr)]
# estimators = [('bar',bar),('rfr',rfr),('gbr',gbr)]
sr = StackingRegressor(estimators=estimators,final_estimator=rfr)
sr.fit(X_train, y_train)

sr_train_score = sr.score(X_train, y_train)
sr_test_score = sr.score(X_test, y_test)
sr_train_predict = sr.predict(X_train)
sr_test_predict = sr.predict(X_test)
# Min blended results
def min_blended_predictions(y_test, sr_test_predict, lr_test_predict, rr_test_predict,
                           lar_test_predict, bar_test_predict, hr_test_predict, rfr_test_predict, abr_test_predict,
                           xgbr_test_predict, gbr_test_predict):
    blended_prediction = []
    diff_list_index = []
    diff1 = abs(sr_test_predict)
    diff2 = abs(lr_test_predict)
    diff3 = abs(rr_test_predict)
    diff4 = abs(lar_test_predict)
    diff5 = abs(bar_test_predict)
    diff6 = abs(hr_test_predict)
    diff7 = abs(rfr_test_predict)
    diff8 = abs(abr_test_predict)
    diff9 = abs(xgbr_test_predict)
    diff10 = abs(gbr_test_predict)
    for i in range(0, len(diff1)):
        diff_list = [diff1[i],diff2[i],diff3[i],diff4[i],diff5[i],diff6[i],diff7[i],diff8[i],diff9[i],diff10[i]]
        diff_index = diff_list.index(min(diff_list))
        if(diff_index == 0):
            blended_prediction.append(sr_test_predict[i])
        elif(diff_index == 1):
            blended_prediction.append(lr_test_predict[i])
        elif(diff_index == 2):
            blended_prediction.append(rr_test_predict[i])
        elif(diff_index == 3):
            blended_prediction.append(lar_test_predict[i])
        elif(diff_index == 4):
            blended_prediction.append(bar_test_predict[i])
        elif(diff_index == 5):
            blended_prediction.append(hr_test_predict[i])
        elif(diff_index == 6):
            blended_prediction.append(rfr_test_predict[i])
        elif(diff_index == 7):
            blended_prediction.append(abr_test_predict[i])
        elif(diff_index == 8):
            blended_prediction.append(xgbr_test_predict[i])
        elif(diff_index == 9):
            blended_prediction.append(gbr_test_predict[i])
        diff_list_index.append(diff_index)
    return blended_prediction
mb_train_predict = min_blended_predictions(y_train, sr_train_predict, lr_train_predict, rr_train_predict,
                           lar_train_predict, bar_train_predict, hr_train_predict, rfr_train_predict, abr_train_predict,
                           xgbr_train_predict, gbr_train_predict)

mb_test_predict = min_blended_predictions(y_test, sr_test_predict, lr_test_predict, rr_test_predict,
                           lar_test_predict, bar_test_predict, hr_test_predict, rfr_test_predict, abr_test_predict,
                           xgbr_test_predict, gbr_test_predict)
print("Train RMSLE Blended", np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(mb_train_predict))))
print("Test RMSLE Blended", np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(mb_test_predict))))
regressor_comparison_df = pd.DataFrame({'metrics': ['Train Score', 'Test Score', 'Train RMSE','Test RMSE','Train MSLE','Test MSLE','Train RMSLE', 'Test RMSLE'],
                             'Stacking Regression':[sr_train_score, sr_test_score, mean_squared_error(np.expm1(y_train), np.expm1(sr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(sr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(sr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(sr_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(sr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(sr_test_predict)))],
                             'Linear Regression':[lr_train_score, lr_test_score, mean_squared_error(np.expm1(y_train), np.expm1(lr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(lr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(lr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(lr_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(lr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(lr_test_predict)))],
                             'Ridge Regression':[rr_train_score, rr_test_score, mean_squared_error(np.expm1(y_train), np.expm1(rr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(rr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(rr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(rr_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(rr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(rr_test_predict)))],
                             'Lasso Regression':[lar_train_score, lar_test_score,mean_squared_error(np.expm1(y_train), np.expm1(lar_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(lar_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(lar_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(lar_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(lar_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(lar_test_predict)))],
                             'Bayseian Regression':[bar_train_score, bar_test_score,mean_squared_error(np.expm1(y_train), np.expm1(bar_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(bar_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(bar_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(bar_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(bar_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(bar_test_predict)))],
                             'Huber Regression':[hr_train_score, hr_test_score,mean_squared_error(np.expm1(y_train), np.expm1(hr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(hr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(hr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(hr_test_predict)),np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(hr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(hr_test_predict)))],
                             'Random Forest Regression':[rfr_train_score, rfr_test_score,mean_squared_error(np.expm1(y_train), np.expm1(rfr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(rfr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(rfr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(rfr_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(rfr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(rfr_test_predict)))],
                             'AdaBoost Regression':[abr_train_score, abr_test_score,mean_squared_error(np.expm1(y_train), np.expm1(abr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(abr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(abr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(abr_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(abr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(abr_test_predict)))],
                             'XGB Regression':[xgbr_train_score, xgbr_test_score,mean_squared_error(np.expm1(y_train), np.expm1(xgbr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(xgbr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(xgbr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(xgbr_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(xgbr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(xgbr_test_predict)))],
                             'GB Regression':[gbr_train_score, gbr_test_score,mean_squared_error(np.expm1(y_train), np.expm1(gbr_train_predict), squared=False), mean_squared_error(np.expm1(y_test), np.expm1(gbr_test_predict), squared=False), mean_squared_log_error(np.expm1(y_train), np.expm1(gbr_train_predict)), mean_squared_log_error(np.expm1(y_test), np.expm1(gbr_test_predict)), np.sqrt(mean_squared_log_error(np.expm1(y_train), np.expm1(gbr_train_predict))), np.sqrt(mean_squared_log_error(np.expm1(y_test), np.expm1(gbr_test_predict)))]
                             })
regressor_comparison_df
final_SalePrice_prediction = mb_test_predict
# Display first 10 actual and prediction SalePrice of train and test data for final prediction
pd.concat([np.expm1(y_train[0:10]),
           pd.Series(np.expm1(mb_train_predict[0:10])),
          np.expm1(y_test[0:10]),
          pd.Series(np.expm1(final_SalePrice_prediction[0:10]))], 
          axis=1, 
          keys=["Train Actual Value", "Train Predicted Value", "Test Actual Value", "Test Predicted Value"])
# Update SalePrice column of submission dataframe with final prediction values
submission['SalePrice'] = np.expm1(final_SalePrice_prediction)
submission.head()
# Save the final SalePrice submission file in the output directory of kaggle
submission.to_csv("HousePriceSubmission.csv", index=False)