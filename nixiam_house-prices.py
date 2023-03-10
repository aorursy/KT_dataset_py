import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
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
with open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt") as f:
    contents = f.read()
    print(contents)
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print("-- df_train shape --")
print(df_train.shape)
print("-- df_test shape --")
print(df_test.shape)
df_train.columns
df_test.columns
frames = [df_train, df_test]
df_result = pd.concat(frames, ignore_index=True, sort=False)
print(df_result.shape)
df_result['SalePrice'].tail()
df_result.head()
df_result.info()
nullseries = df_result.isnull().sum().sort_values(ascending=False)
print("-- Columns null and values --")
print(nullseries[nullseries > 0])
print("-- PoolArea values --")
print(df_result['PoolArea'].value_counts())
print("#"*40)
print("-- PoolQC values --")
print(df_result['PoolQC'].value_counts(dropna=False))
print("#"*40)
print("-- PoolQC where PoolArea = 0 --")
print(df_result[df_result['PoolArea'] == 0]['PoolQC'].value_counts(dropna=False))
print("#"*40)
print("-- PoolQC where PoolArea != 0 --")
print(df_result[df_result['PoolArea'] != 0]['PoolQC'].value_counts(dropna=False))
print("-- PoolQC refill and update values --")
df_result['PoolQC'] = df_result.apply(lambda x: "NA" if pd.isnull(x['PoolQC']) and x["PoolArea"] is 0 else ("TA" if pd.isnull(x['PoolQC']) else x['PoolQC']), axis="columns")
print(df_result['PoolQC'].value_counts())
print("-- MiscFeature removed: Miscellaneous and too many missing values --")
df_result = df_result.drop(['MiscFeature'], axis=1)
print("-- Street values --")
print(df_result['Street'].value_counts(dropna=False))
print("-- Alley values --")
print(df_result['Alley'].value_counts(dropna=False))
print("-- Alley NaN to Alley Pave -- ")
df_result['Alley'].fillna("Pave", inplace=True)
print(df_result['Alley'].value_counts(dropna=False))
print("-- Fence values --")
print(df_result['Fence'].value_counts(dropna=False))
print("-- Fence NaN to Fence NA --")
df_result['Fence'].fillna("NA", inplace=True)
print(df_result['Fence'].value_counts(dropna=False))
print("-- Fireplaces values--")
print(df_result['Fireplaces'].value_counts(dropna=False))
print("-- FireplaceQu values --")
print(df_result['FireplaceQu'].value_counts(dropna=False))
print("-- FireplaceQu NaN to FireplaceQu 0 where Fireplaces is 0 --")
df_result['FireplaceQu'] = df_result.apply(lambda x: "NA" if pd.isnull(x['FireplaceQu']) and x['Fireplaces'] is 0 else x['FireplaceQu'],axis='columns')
print(df_result['FireplaceQu'].value_counts(dropna=False))
print("-- LotFrontage describe --")
print(df_result['LotFrontage'].describe())
print("-- Check where LotFrontage is null related to MSSubClass class --")
print(df_result[df_result['LotFrontage'].isnull()]['MSSubClass'].value_counts(dropna=False))
print("#"*80)
print("-- Check where LotFrontage is null related to MSZoning class --")
print(df_result[df_result['LotFrontage'].isnull()]['MSZoning'].value_counts(dropna=False))
print("-- Describe LotFrontage grouped by MSSubClass class --")
print(df_result.groupby(['MSSubClass'])['LotFrontage'].describe())
print("#"*80)
print("-- Describe LotFrontage grouped by MSZoning class --")
print(df_result.groupby(['MSZoning'])['LotFrontage'].describe())
print("-- Mean of LotFrontage grouped by MSSubClass and MSZoning classes --")
print(df_result.groupby(['MSSubClass','MSZoning'])['LotFrontage'].mean())
print("-- Fillna LotFrontage with mean of classes MSSubClass and MSZoning grouped --")
df_result['LotFrontage'] = df_result['LotFrontage'].fillna(df_result.groupby(['MSSubClass','MSZoning'])['LotFrontage'].transform('mean'))
print("-- Residual NaN values: ", df_result['LotFrontage'].isnull().sum())
print("#"*80)
print("Head 'MSSubClass','MSZoning','LotFrontage")
print(df_result[['MSSubClass','MSZoning','LotFrontage']].head(10))
print(" -- 2 NaN to handler --")
print("-- Check where LotFrontage is null related to MSSubClass class --")
print(df_result[df_result['LotFrontage'].isnull()]['MSSubClass'].value_counts(dropna=False))
print("#"*80)
print("-- Check where LotFrontage is null related to MSZoning class --")
print(df_result[df_result['LotFrontage'].isnull()]['MSZoning'].value_counts(dropna=False))
print("-- Fillna LotFrontage with mean of classes MSSubClass and MSZoning grouped --")
df_result['LotFrontage'] = df_result['LotFrontage'].fillna(df_result.groupby(['MSSubClass'])['LotFrontage'].transform('mean'))
df_result['LotFrontage'] = df_result['LotFrontage'].fillna(df_result.groupby(['MSZoning'])['LotFrontage'].transform('mean'))
print("-- Residual NaN values: ", df_result['LotFrontage'].isnull().sum())
print("#"*80)
print("Head 'MSSubClass','MSZoning','LotFrontage")
print(df_result[['MSSubClass','MSZoning','LotFrontage']].head(10))
print("-- Check how many values are null for the other Garage columns where GarageType is also --")
print(df_result[df_result['GarageType'].isnull()][['GarageType','GarageCond','GarageYrBlt', 'GarageFinish', 'GarageQual']].isnull().sum())
print("-- Fillna GarageType - GarageCond - GarageFinish - GarageQual with NA class --")
print("-- GarageType NaN to GarageType NA --")
df_result['GarageType'].fillna("NA", inplace=True)
print(df_result['GarageType'].value_counts(dropna=False))
print("-- GarageCond NaN to GarageCond NA --")
df_result['GarageCond'].fillna("NA", inplace=True)
print(df_result['GarageCond'].value_counts(dropna=False))
print("-- GarageFinish NaN to GarageFinish NA --")
df_result['GarageFinish'].fillna("NA", inplace=True)
print(df_result['GarageFinish'].value_counts(dropna=False))
print("-- GarageQual NaN to GarageQual NA --")
df_result['GarageQual'].fillna("NA", inplace=True)
print(df_result['GarageQual'].value_counts(dropna=False))
print("Min Year: ", df_result['GarageYrBlt'].min())
print("Max Year: ", df_result['GarageYrBlt'].max())
year_classes = 11
print("Years per classes:" ,(int(df_result['GarageYrBlt'].max()) - int(df_result['GarageYrBlt'].min())) / year_classes)
years_labels = np.arange(start=1, stop=year_classes+1)
print("years_lables: ", years_labels)
GarageYrBlt_classes = pd.qcut(df_result['GarageYrBlt'].rank(method='first'), year_classes, labels=years_labels)
df_result['GarageYrBlt_class'] = GarageYrBlt_classes
print(df_result[['GarageYrBlt', 'GarageYrBlt_class']].head(10))
print(df_result['GarageYrBlt_class'].dtype)
print("-- Check class where GarageYrBlt is NaN --")
print(df_result[df_result['GarageYrBlt'].isnull()][['GarageYrBlt', 'GarageYrBlt_class']].head(10))
print("-- Fill NaN in GarageYrBlt_class with 0 class")
df_result['GarageYrBlt_class'] = df_result['GarageYrBlt_class'].cat.add_categories('0')
df_result['GarageYrBlt_class'].fillna('0', inplace =True)
print(df_result[df_result['GarageYrBlt'].isnull()][['GarageYrBlt', 'GarageYrBlt_class']].head(10))
print("-- Convert to string --")
df_result["GarageYrBlt_class"] = df_result["GarageYrBlt_class"].astype(str)
print(df_result['GarageYrBlt_class'].dtype)
print(df_result[['GarageYrBlt', 'GarageYrBlt_class']].head(10))
print("-- Summary year per classes --")
print(df_result.groupby('GarageYrBlt_class')['GarageYrBlt_class'].value_counts(dropna=False))
print("-- Delete original GarageYrBlt column --")
df_result = df_result.drop('GarageYrBlt', axis=1)
#print("-- Replace GarageYrBlt NaN with 0 --")
#df_result['GarageYrBlt'].fillna(0, inplace=True)
#print("GarageYrBlt with null values: ", df_result['GarageYrBlt'].isnull().sum())
print("-- Check NaN value where BsmtExposure is null --")
print(df_result[df_result['BsmtExposure'].isnull()][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum())
print("-- Check single case null vs not null (38 vs 37)")
print(df_result[df_result['BsmtExposure'].isnull() & df_result['BsmtQual'].notnull()][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].head())
print("-- Value counts for BsmtExposure with BsmtQual == \"Gd\"")
print(df_result[df_result['BsmtQual'] == "Gd"]['BsmtExposure'].value_counts(dropna=False))
print("-- Fill BsmtExposure with \"Av\" for BsmtQual == \"Gd\"")
df_result.loc[df_result['BsmtExposure'].isnull() & df_result['BsmtQual'].notnull(), 'BsmtExposure'] = "Av"
print("-- Value counts for BsmtExposure with BsmtQual == \"Gd\"")
print(df_result[df_result['BsmtQual'] == "Gd"]['BsmtExposure'].value_counts(dropna=False))
print("-- Re-check NaN value where BsmtExposure is null --")
print(df_result[df_result['BsmtExposure'].isnull()][['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum())
print("-- Get Index of row where BsmtExposure is null --")
ind_0 = df_result.index[df_result['BsmtExposure'].isnull() & df_result['BsmtQual'].isnull()].tolist() 
print(ind_0)
print("-- Get Index of row where BsmtExposure and BsmtQual are null --")
ind_1 = df_result.index[df_result['BsmtExposure'].isnull() & df_result['BsmtQual'].isnull()].tolist()
print(ind_1)
print("-- Get Index of row where BsmtExposure and BsmtCond are null --")
ind_2 = df_result.index[df_result['BsmtExposure'].isnull() & df_result['BsmtCond'].isnull()].tolist()
print(ind_2)
print("-- Get Index of row where BsmtExposure and BsmtFinType1 are null --")
ind_3 = df_result.index[df_result['BsmtExposure'].isnull() & df_result['BsmtFinType1'].isnull()].tolist()
print(ind_3)
print("-- Get Index of row where BsmtExposure and BsmtFinType2 are null --")
ind_4 = df_result.index[df_result['BsmtExposure'].isnull() & df_result['BsmtFinType2'].isnull()].tolist()
print(ind_4)
print("-- Compare the arrays --")
compare = ind_0 == ind_1 == ind_2 == ind_3 == ind_4
print("Array have same indexes: ", compare)
print("-- Fill BsmtQual - BsmtCond - BsmtExposure - BsmtFinType1 - BsmtFinType2 with \"NA\" ")
df_result['BsmtQual'].fillna("NA", inplace=True)
df_result['BsmtExposure'].fillna("NA", inplace=True)
df_result['BsmtCond'].fillna("NA", inplace=True)
df_result['BsmtFinType1'].fillna("NA", inplace=True)
df_result['BsmtFinType2'].fillna("NA", inplace=True)
print("-- Re-check NaN values --")
print(df_result[['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']].isnull().sum())
print("-- Get MasVnrType value counts --")
print(df_result['MasVnrType'].value_counts(dropna=False))
print("-- Get MasVnrArea describe --")
print(df_result['MasVnrArea'].describe())
print("-- Get MasVnrArea value counts --")
print(df_result.groupby('MasVnrType')['MasVnrArea'].describe())
print("-- Get index of MasVnrType and MasVnrArea where they are NaN --")
ind_0 = df_result.index[df_result['MasVnrType'].isnull()].tolist()
print(ind_0)
ind_1 = df_result.index[df_result['MasVnrArea'].isnull()].tolist()
print(ind_1)
print("-- Compare array of indexes --")
compare = ind_0 == ind_1
print("Array have same indexes: ", compare)
print("-- Fill MasVnrType with \"None\"")
df_result['MasVnrType'].fillna("None", inplace=True)
print("-- Fill MasVnrType with 0")
df_result['MasVnrArea'].fillna(0, inplace=True)
print("-- Ri-check NaN values for MasVnrType and MasVnrType")
print(df_result[['MasVnrType', 'MasVnrArea']].isnull().sum())
print("-- Get MSZoning value counts include NaN values --")
print(df_result['MSZoning'].value_counts(dropna=False))
print("-- View MSZoning values groupby MSSubClass --")
print(df_result.groupby('MSSubClass')['MSZoning'].value_counts(dropna=False))
print("-- fillna MSZoning category with most popular in MSSubClass group --")
df_result.loc[(df_result['MSZoning'].isnull()) & (df_result['MSSubClass'] == 20), 'MSZoning'] = "RL"
df_result.loc[(df_result['MSZoning'].isnull()) & (df_result['MSSubClass'] == 30), 'MSZoning'] = "RL"
df_result.loc[(df_result['MSZoning'].isnull()) & (df_result['MSSubClass'] == 70), 'MSZoning'] = "RL"
print("-- Get MSZoning value counts include NaN values --")
print(df_result['MSZoning'].value_counts(dropna=False))
print("#"*60)
print("-- View MSZoning values groupby MSSubClass --")
print(df_result.groupby('MSSubClass')['MSZoning'].value_counts(dropna=False))
print("-- Get Utilities value counts include NaN values --")
print(df_result['Utilities'].value_counts(dropna=False))
print("-- fillna with AllPub value --")
df_result['Utilities'].fillna("AllPub", inplace=True)
print("-- Get Utilities value counts include NaN values --")
print(df_result['Utilities'].value_counts(dropna=False))
print("-- Get Functional value counts include NaN values --")
print(df_result['Functional'].value_counts(dropna=False))
print("-- fillna with Typ value --")
df_result['Functional'].fillna("Typ", inplace=True)
print("-- Get Functional value counts include NaN values --")
print(df_result['Functional'].value_counts(dropna=False))
print("-- Get BsmtFullBath value counts include NaN values --")
print(df_result['BsmtFullBath'].value_counts(dropna=False))
print("#"*60)
print("-- Get BsmtHalfBath value counts include NaN values --")
print(df_result['BsmtHalfBath'].value_counts(dropna=False))
print("-- Get BsmtFullBath - BsmtHalfBath and BsmtHalfBath details when NaN --")
print(df_result[df_result['BsmtFullBath'].isnull() | df_result['BsmtHalfBath'].isnull()][['BsmtQual','BsmtFullBath', 'BsmtHalfBath']])
print("-- fillna BsmtFullBath and  BsmtHalfBath with 0.0 --")
df_result['BsmtFullBath'].fillna(0.0, inplace=True)
df_result['BsmtHalfBath'].fillna(0.0, inplace=True)
print("-- Ri-check BsmtFullBath value counts include NaN values --")
print(df_result['BsmtFullBath'].value_counts(dropna=False))
print("#"*60)
print("-- Ri-check BsmtHalfBath value counts include NaN values --")
print(df_result['BsmtHalfBath'].value_counts(dropna=False))
print("-- Get GarageArea value counts include NaN values --")
print(df_result['GarageArea'].isnull().sum())
print("#"*60)
print("-- Get GarageArea value counts include NaN values --")
print(df_result['GarageCars'].isnull().sum())
print("-- get GarageArea grouped by GarageType --")
print(df_result[df_result['GarageArea'].isnull()][['GarageType','LotArea','GarageArea', 'GarageCars']])
print("-- Print descrive of 'LotArea','GarageArea','GarageCars' where GarageType is Detchd")
print(df_result[df_result['GarageType'] == "Detchd"][['LotArea','GarageArea','GarageCars']].describe())
print("-- fillna GarageArea with 500 --")
df_result['GarageArea'].fillna(500.0, inplace=True)
print("-- fillna GarageCars with 2 --")
df_result['GarageCars'].fillna(2.0, inplace=True)
print("-- Richeck GarageArea value counts include NaN values --")
print(df_result['GarageArea'].isnull().sum())
print("#"*60)
print("-- Richeck GarageArea value counts include NaN values --")
print(df_result['GarageCars'].isnull().sum())
print("-- Get BsmtFinSF2 value counts include NaN values --")
print(df_result['BsmtFinSF2'].isnull().sum())
print("-- Describe BsmtFinSF2")
print(df_result['BsmtFinSF2'].describe())
print("-- Get 'BsmtFinType2','BsmtFinSF2' where BsmtFinSF2 is NA")
print(df_result[df_result['BsmtFinSF2'].isnull()][['BsmtFinType2','BsmtFinSF2']])
print("-- Describe BsmtFinSF2 group by BsmtFinType2 --")
print(df_result[df_result['BsmtFinType2'] == 'NA']['BsmtFinSF2'].mode())
print("-- fillna BsmtFinSF2 with 0 --")
df_result['BsmtFinSF2'].fillna(0.0, inplace=True)
print("-- Ri-check BsmtFinSF2 value counts include NaN values --")
print(df_result['BsmtFinSF2'].isnull().sum())
print("-- Get Exterior1st value counts include NaN values --")
print(df_result['Exterior1st'].value_counts(dropna=False))
print("#"*60)
print("-- Get Exterior2nd value counts include NaN values --")
print(df_result['Exterior2nd'].value_counts(dropna=False))
print("-- Check if 'Exterior1st', 'Exterior2nd'are NaN at the same index of row --")
print(df_result[df_result['Exterior1st'].isnull()][['Exterior1st', 'Exterior2nd']])
print("-- Get MasVnrType value where Exterior1st is NaN --")
print(df_result[df_result['Exterior1st'].isnull()]['MasVnrType'].value_counts(dropna=False))
print("-- Get Exterior1st where Exterior1st is None --")
print(df_result[df_result['MasVnrType'] == "None"]['Exterior1st'].value_counts(dropna=False))
print("#"*60)
print("-- Get Exterior2nd where Exterior1st is None --")
print(df_result[df_result['MasVnrType'] == "None"]['Exterior2nd'].value_counts(dropna=False))
print("-- Fill Exterior1st and Exterior2nd with VinylSd class")
df_result['Exterior1st'].fillna("VinylSd", inplace=True)
df_result['Exterior2nd'].fillna("VinylSd", inplace=True)
print("-- Ri-Check NaN values for Exterior1st and Exterior2nd")
print(df_result['Exterior1st'].isnull().sum())
print(df_result['Exterior2nd'].isnull().sum())
print("-- Get BsmtUnfSF value counts include NaN values --")
print(df_result['BsmtUnfSF'].isnull().sum())
print("-- Describe BsmtUnfSF")
print(df_result['BsmtUnfSF'].describe())
print("-- Get BsmtUnfSF where BsmtQual is NA --")
print(df_result[df_result['BsmtQual'] == "NA"]['BsmtUnfSF'].isnull().sum())
print("-- Get BsmtUnfSF where BsmtQual is NA --")
print(df_result[df_result['BsmtQual'] == "NA"]['BsmtUnfSF'].describe())
print("-- Fill BsmtUnfSF with 0")
df_result['BsmtUnfSF'].fillna(0.0, inplace=True)
print("-- Get BsmtUnfSF value counts include NaN values --")
print(df_result['BsmtUnfSF'].isnull().sum())
print("-- Get TotalBsmtSF value counts include NaN values --")
print(df_result['TotalBsmtSF'].isnull().sum())
print("-- Describe TotalBsmtSF")
print(df_result['TotalBsmtSF'].describe())
print("-- Get TotalBsmtSF where BsmtQual is NA --")
print(df_result[df_result['BsmtQual'] == "NA"]['TotalBsmtSF'].isnull().sum())
print("-- Fill TotalBsmtSF with 0")
df_result['TotalBsmtSF'].fillna(0.0, inplace=True)
print("-- Get TotalBsmtSF value counts include NaN values --")
print(df_result['TotalBsmtSF'].isnull().sum())
print("-- Get BsmtFinSF1 value counts include NaN values --")
print(df_result['BsmtFinSF1'].isnull().sum())
print("-- Describe BsmtFinSF1")
print(df_result['BsmtFinSF1'].describe())
print("-- Get BsmtFinSF1 where BsmtQual is NA --")
print(df_result[df_result['BsmtQual'] == "NA"]['BsmtFinSF1'].isnull().sum())
print("-- Fill BsmtFinSF1 with 0")
df_result['BsmtFinSF1'].fillna(0.0, inplace=True)
print("-- Get BsmtFinSF1 value counts include NaN values --")
print(df_result['BsmtFinSF1'].isnull().sum())
print("-- Check NaN values for Electrical")
print(df_result['Electrical'].isnull().sum())
print("-- Get value counts for Electrical")
print(df_result['Electrical'].value_counts(dropna=False))
print("-- Fill Electrical with \"SBrkr\" - Stantard value")
df_result['Electrical'].fillna("SBrkr", inplace=True)
print("-- Ri-Check NaN values for Electrical")
print(df_result['Electrical'].isnull().sum())
print("-- Get KitchenQual value counts with NaN --")
print(df_result['KitchenQual'].value_counts(dropna=False))
print("-- Fill KitchenQual with \"TA\" - Stantard value")
df_result['KitchenQual'].fillna("TA", inplace=True)
print("-- Ri-Check NaN values for KitchenQual")
print(df_result['KitchenQual'].isnull().sum())
print("-- Get SaleType value counts with NaN --")
print(df_result['SaleType'].value_counts(dropna=False))
print("-- Get SaleType values groupped by SaleCondition")
print(df_result.groupby('SaleCondition')['SaleType'].value_counts(dropna=False))
print("-- Fill SaleType with \"WD\" - Stantard value")
df_result['SaleType'].fillna("WD", inplace=True)
print("-- Ri-Check NaN values for SaleType")
print(df_result['SaleType'].isnull().sum())
nullseries = df_result.isnull().sum().sort_values(ascending=False)
print("-- Columns null and values --")
print(nullseries[nullseries > 0])
print("-- Get columns with data type equal to Int --")
print(df_result.select_dtypes(include=['int']).columns)
df_result["MSSubClass"] = df_result["MSSubClass"].astype(str)
df_result["LotArea"] = df_result["LotArea"].astype(str)
df_result["OverallQual"] = df_result["OverallQual"].astype(str)
df_result["OverallCond"] = df_result["OverallCond"].astype(str)
print("-- ri-check: Get columns with data type equal to Int --")
print(df_result.select_dtypes(include=['int']).columns)
print("-- Check 'YearBuilt' in details --")
print(f"Min:{df_result['YearBuilt'].min()} - Max:{df_result['YearBuilt'].max()}")
print("Describe:")
print(df_result['YearBuilt'].describe())
print("-"*40)
print("-- Check 'YearRemodAdd' in details --")
print(f"Min:{df_result['YearRemodAdd'].min()} - Max:{df_result['YearRemodAdd'].max()}")
print("Describe:")
print(df_result['YearRemodAdd'].describe())
print("-"*40)
print("-- Check 'YrSold' in details --")
print(f"Min:{df_result['YrSold'].min()} - Max:{df_result['YrSold'].max()}")
print("Describe:")
print(df_result['YrSold'].describe())
YearBuilt_labels = np.arange(start=1, stop=year_classes+1)
print("years_lables: ", YearBuilt_labels)
YearBuilt_class = pd.qcut(df_result['YearBuilt'].rank(method='first'), year_classes, labels=YearBuilt_labels)
df_result['YearBuilt_class'] = GarageYrBlt_classes
print(df_result[['YearBuilt', 'YearBuilt_class']].head(10))
df_result["YearBuilt_class"] = df_result["YearBuilt_class"].astype(str)
print(df_result['YearBuilt_class'].dtype)
YearRemodAdd_labels = np.arange(start=1, stop=year_classes+1)
print("years_lables: ", YearRemodAdd_labels)
YearRemodAdd_class = pd.qcut(df_result['YearRemodAdd'].rank(method='first'), year_classes, labels=YearRemodAdd_labels)
df_result['YearRemodAdd_class'] = YearRemodAdd_class
print(df_result[['YearRemodAdd', 'YearRemodAdd_class']].head(10))
df_result["YearRemodAdd_class"] = df_result["YearRemodAdd_class"].astype(str)
print(df_result['YearRemodAdd_class'].dtype)
YrSold_labels = np.arange(start=1, stop=year_classes+1)
print("years_lables: ", YrSold_labels)
YrSold_class = pd.qcut(df_result['YrSold'].rank(method='first'), year_classes, labels=YrSold_labels)
df_result['YrSold_class'] = GarageYrBlt_classes
print(df_result[['YrSold', 'YrSold_class']].head(10))
df_result["YrSold_class"] = df_result["YrSold_class"].astype(str)
print(df_result['YrSold_class'].dtype)
print("-- Delete original Year columns --")
df_result = df_result.drop(['YearBuilt', 'YearRemodAdd', 'YrSold'], axis="columns")
print("-- Get columns with data type equal to Int --")
print(df_result.select_dtypes(include=['float']).columns)
print("-- Get columns with data type equal to Int --")
print(df_result.select_dtypes(include=['category']).columns)
print("-- Get columns with data type equal to Int --")
print(df_result.select_dtypes(include=['object']).columns)
print("-- LotArea head --")
print(df_result['LotArea'].head())
print("-- Cast type to Int")
df_result['LotArea'] = df_result['LotArea'].astype(int)
print(df_result['LotArea'].dtype)
#print("-- df_result shape --")
#print(df_result.shape)
#print("-- re-split df_result to dataset train and test -- ")
#df_train = df_result[:1460]
#df_test = df_result[1460:]
#print("-- df_train shape --")
#print(df_train.shape)
#print("-- df_test shape --")
#print(df_test.shape)
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:29:16 2020

@author: benedet
"""

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
class DataFrameEncoder(TransformerMixin):

    def __init__(self):
        """Encode the data.

        Columns of data type object are appended in the list. After 
        appending Each Column of type object are taken dummies and 
        successively removed and two Dataframes are concated again.

        """
    def fit(self, X, y=None):
        self.object_col = []
        self.hotencoders_list = {}
        for col in X.columns:
            #print(col, X[col].dtype)
            if(X[col].dtype == np.dtype('O') or X[col].dtype == np.dtype('str')):
                self.object_col.append(col)
                colum = np.array(X[col]).reshape(-1, 1)
                #print(colum, colum.dtype)
                self.hotencoders_list[col] = OneHotEncoder(handle_unknown='ignore').fit(colum)
        return self

    def transform(self, X, y=None, drop = True):
        for col in self.object_col:
            #print(col)
            colum = np.array(X[col]).reshape(-1, 1)
            #print(colum, self.hotencoders_list[col].categories_)
            tmp = self.hotencoders_list[col].transform(colum).toarray()
            if drop:
                tmp = pd.DataFrame(tmp[:, 1:])
            else:
                tmp = pd.DataFrame(tmp)
            X = X.drop(col,axis=1)
            X = pd.concat([tmp,X],axis=1)
        return X
    
    def inverse_transform(self, X, y=None, drop = True):
        out = pd.DataFrame(columns=self.object_col)
        counter = 0
        for col in self.object_col:
            cat = len(self.hotencoders_list[col].categories_) + 1
            tmp = self.hotencoders_list[col].inverse_transform(X[counter:cat])
            tmp = pd.DataFrame(tmp)
            out = pd.concat([tmp,out],axis=1)
            counter = counter + 1
        return out
X_tot = df_result.loc[:, df_result.columns != 'SalePrice']
Y_tot = df_result.loc[:, 'SalePrice'].values
Y_tot = Y_tot.reshape(-1, 1)
print(f"X_tot shape:{X_tot.shape} - Y_tot shape:{Y_tot.shape}")
de = DataFrameEncoder().fit(X_tot)
X_tot = de.transform(X_tot)
print(f"X shape:{X_tot.shape}")
X = X_tot[:1460]
Y = Y_tot[:1460]
print(f"X shape:{X.shape} - Y shape:{Y.shape}")
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

X = scaler_X.fit_transform(X)
Y = scaler_y.fit_transform(Y)
print("-- Splitting the dataset into the Training set and Val set --")
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size = 0.1, random_state = 42)

#X_train = X
#y_train = Y

print(f"X_train shape:{X_train.shape} - y_train shape:{y_train.shape}")
print(f"X_val shape:{X_val.shape} - y_val shape:{y_val.shape}")
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam

lr = 0.0005
adam = Adam(lr)

input_data = Input(shape=(X_train.shape[1],))

x = Dense(name="Dense_1", units=32, activation='relu')(input_data)
x = Dense(name="Dense_2", units=32, activation='relu')(x)
x = Dropout(0.1)(x)
x = Dense(name="Dense_3", units=32, activation='relu')(x)
o = Dense(units=y_train.shape[1])(x)

model = Model(inputs=[input_data], outputs=[o])
model.compile(optimizer=adam, loss='mse', metrics=['mse'])
model.summary()
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), verbose=1)
def visualize_learning_curve(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['mse'])
    plt.plot(history.history['val_mse'])
    plt.title('model mse')
    plt.ylabel('mse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
visualize_learning_curve(history)
df_test.head(2)
X_test = X_tot[1460:]
X_test = X_test.loc[:, X_test.columns != 'SalePrice']
print(X_test.shape)
X_test = scaler_X.transform(X_test)
y_pred = model.predict(X_test)
y_pred[0]
y_pred = scaler_y.inverse_transform(y_pred)
y_pred[0]
y_pred.shape
df_sub = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
print(df_sub.shape)
df_sub.head(5)
df_sub['SalePrice'] = y_pred.flatten()
df_sub.head(5)
df_sub.to_csv('submission.csv', index=False)