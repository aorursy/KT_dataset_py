# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.



# Part 1 - Preprocessing the data

# Importing the dataset
dataset_train = pd.read_csv('../input/train.csv')
dataset_test = pd.read_csv('../input/test.csv')
dataset = [dataset_train, dataset_test]
dataset = pd.concat(dataset)

# Making Backward Elimination to find only variables with good correlation for Sales Price (p-value<0.05)

# Filling nan values 

# Filling nan values Alley
dataset.Alley.fillna("No info", inplace = True)

# Filling nan values FireplaceQu
dataset.FireplaceQu.fillna(dataset.FireplaceQu.value_counts().idxmax(), inplace = True)

# Filling nan values FireplaceQu
dataset.FireplaceQu.fillna(dataset.FireplaceQu.value_counts().idxmax(), inplace = True)

# Filling nan values MSZoning
dataset.MSZoning.fillna(dataset.MSZoning.value_counts().idxmax(), inplace = True)

# Filling nan values MasVnrType
dataset.MasVnrType.fillna(dataset.MasVnrType.value_counts().idxmax(), inplace = True)

# Filling nan values MasVnrArea
dataset['MasVnrArea'] = dataset['MasVnrArea'].fillna(dataset['MasVnrArea'].mean() )

# Filling nan values Exterior2nd
dataset.Exterior2nd.fillna(dataset.Exterior2nd.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtQual
dataset.BsmtQual.fillna(dataset.BsmtQual.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtCond
dataset.BsmtCond.fillna(dataset.BsmtCond.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtExposure
dataset.BsmtExposure.fillna(dataset.BsmtExposure.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtFinType1
dataset.BsmtFinType1.fillna(dataset.BsmtFinType1.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtFinType2
dataset.BsmtFinType2.fillna(dataset.BsmtFinType2.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtFullBath
dataset.BsmtFullBath.fillna(dataset.BsmtFullBath.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtHalfBath
dataset.BsmtHalfBath.fillna(dataset.BsmtHalfBath.value_counts().idxmax(), inplace = True)

# Filling nan values Electrical
dataset.Electrical.fillna(dataset.Electrical.value_counts().idxmax(), inplace = True)

# Filling nan values BsmtFinSF1
dataset['BsmtFinSF1'] = dataset['BsmtFinSF1'].fillna(dataset['BsmtFinSF1'].mean() )

# Filling nan values BsmtFinSF2
dataset['BsmtFinSF2'] = dataset['BsmtFinSF2'].fillna(dataset['BsmtFinSF2'].mean() )

# Filling nan values BsmtUnfSF 
dataset['BsmtUnfSF'] = dataset['BsmtUnfSF'].fillna(dataset['BsmtUnfSF'].mean() )

# Filling nan values TotalBsmtSF 
dataset['TotalBsmtSF'] = dataset['TotalBsmtSF'].fillna(dataset['TotalBsmtSF'].mean() )

# Filling nan values KitchenQual 
dataset.KitchenQual.fillna(dataset.KitchenQual.value_counts().idxmax(), inplace = True)

# Filling nan values Functional 
dataset.Functional.fillna(dataset.Functional.value_counts().idxmax(), inplace = True)

# Filling nan values GarageFinish
dataset.GarageFinish.fillna(dataset.GarageFinish.value_counts().idxmax(), inplace = True)

# Filling nan values GarageCars 
dataset['GarageCars'] = dataset['GarageCars'].fillna(dataset['GarageCars'].mean() )

# Filling nan values GarageArea 
dataset['GarageArea'] = dataset['GarageArea'].fillna(dataset['GarageArea'].mean() )

# Filling nan values GarageQual
dataset.GarageQual.fillna(dataset.GarageQual.value_counts().idxmax(), inplace = True)

# Filling nan values GarageCond
dataset.GarageCond.fillna(dataset.GarageCond.value_counts().idxmax(), inplace = True)

# Filling nan values Exterior1st
dataset.Exterior1st.fillna(dataset.Exterior1st.value_counts().idxmax(), inplace = True)

# Filling nan values Fence
dataset.Fence.fillna("No info", inplace = True)

# Filling nan values GarageType
dataset.GarageType.fillna(dataset.GarageType.value_counts().idxmax(), inplace = True)

# Correcting year 2207 and Filling nan values GarageYrBlt
dataset.GarageYrBlt = dataset.GarageYrBlt.replace(2207,2007)
dataset.GarageYrBlt.fillna(dataset.GarageYrBlt.value_counts().idxmax(), inplace = True)

# Filling nan values LotFrontage 
dataset['LotFrontage'] = dataset['LotFrontage'].fillna(dataset['LotFrontage'].mean() )

# Filling nan values SaleType
dataset.SaleType.fillna(dataset.SaleType.value_counts().idxmax(), inplace = True)

# Filling nan values Utilities
dataset.Utilities.fillna(dataset.Utilities.value_counts().idxmax(), inplace = True)

# Filling nan values MiscFeature
dataset.MiscFeature.fillna("No info", inplace = True)

# Filling nan values PoolQC
dataset.PoolQC.fillna("No info", inplace = True)

dataset.info()


# Enconding Categorical Data
from sklearn.preprocessing import LabelEncoder

MSZoning = LabelEncoder()
dataset['MSZoning'] = MSZoning.fit_transform(dataset['MSZoning'])

Street = LabelEncoder()
dataset['Street'] = Street.fit_transform(dataset['Street'])

LandContour = LabelEncoder()
dataset['LandContour'] = LandContour.fit_transform(dataset['LandContour'])

LotConfig = LabelEncoder()
dataset['LotConfig'] = LotConfig.fit_transform(dataset['LotConfig'])

LandSlope = LabelEncoder()
dataset['LandSlope'] = LandSlope.fit_transform(dataset['LandSlope'])

Neighborhood = LabelEncoder()
dataset['Neighborhood'] = Neighborhood.fit_transform(dataset['Neighborhood'])

Condition1 = LabelEncoder()
dataset['Condition1'] = Condition1.fit_transform(dataset['Condition1'])

Condition2 = LabelEncoder()
dataset['Condition2'] = Condition2.fit_transform(dataset['Condition2'])

BldgType = LabelEncoder()
dataset['BldgType'] = BldgType.fit_transform(dataset['BldgType'])

HouseStyle = LabelEncoder()
dataset['HouseStyle'] = HouseStyle.fit_transform(dataset['HouseStyle'])

RoofMatl = LabelEncoder()
dataset['RoofMatl'] = RoofMatl.fit_transform(dataset['RoofMatl'])

Exterior2nd = LabelEncoder()
dataset['Exterior2nd'] = Exterior2nd.fit_transform(dataset['Exterior2nd'])

MasVnrType = LabelEncoder()
dataset['MasVnrType'] = MasVnrType.fit_transform(dataset['MasVnrType'])

ExterQual = LabelEncoder()
dataset['ExterQual'] = ExterQual.fit_transform(dataset['ExterQual'])

Foundation = LabelEncoder()
dataset['Foundation'] = Foundation.fit_transform(dataset['Foundation'])

BsmtQual = LabelEncoder()
dataset['BsmtQual'] = BsmtQual.fit_transform(dataset['BsmtQual'])

BsmtCond = LabelEncoder()
dataset['BsmtCond'] = BsmtCond.fit_transform(dataset['BsmtCond'])

BsmtExposure = LabelEncoder()
dataset['BsmtExposure'] = BsmtExposure.fit_transform(dataset['BsmtExposure'])

BsmtFinType1 = LabelEncoder()
dataset['BsmtFinType1'] = BsmtFinType1.fit_transform(dataset['BsmtFinType1'])

BsmtFinType2 = LabelEncoder()
dataset['BsmtFinType2'] = BsmtFinType2.fit_transform(dataset['BsmtFinType2'])

KitchenQual = LabelEncoder()
dataset['KitchenQual'] = KitchenQual.fit_transform(dataset['KitchenQual'])

Functional = LabelEncoder()
dataset['Functional'] = Functional.fit_transform(dataset['Functional'])

FireplaceQu = LabelEncoder()
dataset['FireplaceQu'] = FireplaceQu.fit_transform(dataset['FireplaceQu'])

GarageFinish = LabelEncoder()
dataset['GarageFinish'] = GarageFinish.fit_transform(dataset['GarageFinish'])

GarageQual = LabelEncoder()
dataset['GarageQual'] = GarageQual.fit_transform(dataset['GarageQual'])

GarageCond = LabelEncoder()
dataset['GarageCond'] = GarageCond.fit_transform(dataset['GarageCond'])

SaleCondition = LabelEncoder()
dataset['SaleCondition'] = SaleCondition.fit_transform(dataset['SaleCondition'])

Alley = LabelEncoder()
dataset['Alley'] = SaleCondition.fit_transform(dataset['Alley'])

CentralAir = LabelEncoder()
dataset['CentralAir'] = SaleCondition.fit_transform(dataset['CentralAir'])

Electrical = LabelEncoder()
dataset['Electrical'] = SaleCondition.fit_transform(dataset['Electrical'])

ExterCond = LabelEncoder()
dataset['ExterCond'] = SaleCondition.fit_transform(dataset['ExterCond'])

Exterior1st = LabelEncoder()
dataset['Exterior1st'] = SaleCondition.fit_transform(dataset['Exterior1st'])

Fence = LabelEncoder()
dataset['Fence'] = SaleCondition.fit_transform(dataset['Fence'])

GarageType = LabelEncoder()
dataset['GarageType'] = SaleCondition.fit_transform(dataset['GarageType'])

Heating = LabelEncoder()
dataset['Heating'] = SaleCondition.fit_transform(dataset['Heating'])

HeatingQC = LabelEncoder()
dataset['HeatingQC'] = SaleCondition.fit_transform(dataset['HeatingQC'])

LotShape = LabelEncoder()
dataset['LotShape'] = SaleCondition.fit_transform(dataset['LotShape'])

MiscFeature = LabelEncoder()
dataset['MiscFeature'] = SaleCondition.fit_transform(dataset['MiscFeature'])

PavedDrive = LabelEncoder()
dataset['PavedDrive'] = SaleCondition.fit_transform(dataset['PavedDrive'])

PoolQC = LabelEncoder()
dataset['PoolQC'] = SaleCondition.fit_transform(dataset['PoolQC'])

RoofStyle = LabelEncoder()
dataset['RoofStyle'] = SaleCondition.fit_transform(dataset['RoofStyle'])

SaleType = LabelEncoder()
dataset['SaleType'] = SaleCondition.fit_transform(dataset['SaleType'])

Utilities = LabelEncoder()
dataset['Utilities'] = SaleCondition.fit_transform(dataset['Utilities'])


# create dummy variables for categorical data and exclude 1 one of them to avoid the dummy variable trap
dataset = pd.get_dummies(dataset, columns=['MSZoning', 'Street', 'LandContour', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofMatl', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'SaleCondition', 'Alley', 'CentralAir', 'Electrical', 'ExterCond', 'Exterior1st', 'Fence', 'GarageType', 'Heating', 'HeatingQC', 'LotShape', 'MiscFeature', 'PavedDrive', 'PoolQC', 'RoofStyle', 'SaleType', 'Utilities'], drop_first=True)


# Updating datasets
dataset_train = dataset.iloc[:1460,:]
dataset_test = dataset.iloc[1460:,:]


# Splitting the dataset into the input and output
X_train = dataset_train.iloc[:,dataset_train.columns != 'Id' ]
X_train = X_train.drop('SalePrice', axis = 1)
y_train = dataset_train.iloc[: , dataset_train.columns == 'SalePrice']
X_test = dataset_test.iloc[:,dataset_train.columns != 'Id' ]
X_test = X_test.drop('SalePrice', axis = 1)

#getting the PassIndex for the Submission dataset
pass_index = dataset.iloc[1460:,17]


# Backward Eliminiation

# Insert B Intercept
X_train['constant'] = 1
X_train = X_train[['constant'] + X_train.columns[:-1].tolist()]


# Call Ordinary Least Square and Clean irrelevant data from X_train
import statsmodels.formula.api as sm

deleted_columns = []

for i in range (0,37):
    xelimination = X_train
    regressorOLS = sm.OLS(y_train, xelimination).fit()

    pvalue = []

    if i < 36:
        for j in range (1,(37 - i)):
            pvalue.append(regressorOLS.pvalues[j])

        if (max(pvalue) > 0.05):
            index = np.argmax(pvalue)
            if (X_train.columns[index] != 'constant'):
                deleted_columns.append(X_train.columns[index+1])
                X_train.drop(X_train.columns[index+1], axis=1, inplace=True)
            else:
                print(index)
                break

regressorOLS.summary()

# As we can see, now we have only relevat data! Now let's visually exclude dummies whose set is entirely higher than 0.05 for p-value 
inicio = X_train.columns.get_loc("Exterior2nd_1")
fim = X_train.columns.get_loc("Exterior2nd_15")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("MasVnrType_1")
fim = X_train.columns.get_loc("MasVnrType_3")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Heating_1")
fim = X_train.columns.get_loc("Heating_5")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("FireplaceQu_1")
fim = X_train.columns.get_loc("FireplaceQu_4")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Alley_1")
fim = X_train.columns.get_loc("Alley_2")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("CentralAir_1")
fim = X_train.columns.get_loc("CentralAir_1")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("GarageFinish_1")
fim = X_train.columns.get_loc("GarageFinish_2")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Electrical_1")
fim = X_train.columns.get_loc("Electrical_4")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("ExterCond_1")
fim = X_train.columns.get_loc("ExterCond_4")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("GarageType_1")
fim = X_train.columns.get_loc("GarageType_5")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("LotShape_1")
fim = X_train.columns.get_loc("LotShape_3")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("MiscFeature_1")
fim = X_train.columns.get_loc("MiscFeature_4")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("PavedDrive_1")
fim = X_train.columns.get_loc("PavedDrive_2")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("SaleType_1")
fim = X_train.columns.get_loc("SaleType_8")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Utilities_1")
fim = X_train.columns.get_loc("Utilities_1")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Foundation_1")
fim = X_train.columns.get_loc("Foundation_5")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("LandContour_1")
fim = X_train.columns.get_loc("LandContour_3")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("HouseStyle_1")
fim = X_train.columns.get_loc("HouseStyle_7")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("PoolQC_1")
fim = X_train.columns.get_loc("PoolQC_3")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Exterior1st_1")
fim = X_train.columns.get_loc("Exterior1st_14")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Fence_1")
fim = X_train.columns.get_loc("Fence_4")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Neighborhood_1")
fim = X_train.columns.get_loc("Neighborhood_24")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Condition1_1")
fim = X_train.columns.get_loc("Condition1_8")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Condition2_1")
fim = X_train.columns.get_loc("Condition2_7")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("RoofMatl_1")
fim = X_train.columns.get_loc("RoofMatl_7")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("Functional_1")
fim = X_train.columns.get_loc("Functional_6")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()

inicio = X_train.columns.get_loc("BsmtFinType2_1")
fim = X_train.columns.get_loc("BsmtFinType2_5")
for i in range(0, (fim - inicio +1)):
    deleted_columns.append(X_train.columns[inicio])
    X_train.drop(X_train.columns[inicio], axis=1, inplace=True)
xelimination = X_train
regressorOLS = sm.OLS(y_train, xelimination).fit()
regressorOLS.summary()




# Cleaning the data
X_test.drop(deleted_columns, axis=1, inplace=True)
X_train.drop(X_train.columns[0], axis=1, inplace=True)



# Part 2 - Making the predictions


from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

# XGBoost model
XGBoost = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

#Gradient Boosting Regression :
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

# LightGBM
LightGBM = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

XGBoost.fit(X_train,y_train.values.ravel())
y_pred_XGBoost = XGBoost.predict(X_test)

GBoost.fit(X_train,y_train.values.ravel())
y_pred_GBoost = GBoost.predict(X_test)

LightGBM.fit(X_train,y_train.values.ravel())
y_pred_LightGBM = LightGBM.predict(X_test)

y_pred = 0.05*y_pred_XGBoost + 0.9*y_pred_GBoost + 0.05*y_pred_LightGBM


#Part 3 - Generating Submission File
only_final_values = pd.DataFrame(columns = ['Id', 'SalePrice'])
only_final_values['Id'] = pass_index
only_final_values['SalePrice'] = y_pred
only_final_values['SalePrice'] = only_final_values['SalePrice'].astype(np.float64)
only_final_values.to_csv('house_prices_submission_final.csv', index = False)


