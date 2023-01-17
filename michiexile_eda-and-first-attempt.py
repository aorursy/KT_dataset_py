%pylab inline
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
data_train = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
from sklearn import pipeline, compose, impute, preprocessing
all_features = ['Id', 'MSSubClass', 'MSZoning', 
                'LotFrontage', 'LotArea', 'Street',
                'Alley', 'LotShape', 'LandContour', 
                'Utilities', 'LotConfig','LandSlope', 
                'Neighborhood', 'Condition1', 'Condition2', 
                'BldgType','HouseStyle', 'OverallQual', 
                'OverallCond', 'YearBuilt', 'YearRemodAdd',
                'RoofStyle', 'RoofMatl', 'Exterior1st', 
                'Exterior2nd', 'MasVnrType','MasVnrArea', 
                'ExterQual', 'ExterCond', 'Foundation', 
                'BsmtQual','BsmtCond', 'BsmtExposure', 
                'BsmtFinType1', 'BsmtFinSF1','BsmtFinType2',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                'Heating','HeatingQC', 'CentralAir', 
                'Electrical', '1stFlrSF', '2ndFlrSF',
                'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                'BsmtHalfBath', 'FullBath','HalfBath', 
                'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
                'TotRmsAbvGrd', 'Functional', 'Fireplaces', 
                'FireplaceQu', 'GarageType','GarageYrBlt', 
                'GarageFinish', 'GarageCars', 'GarageArea', 
                'GarageQual','GarageCond', 'PavedDrive', 
                'WoodDeckSF', 'OpenPorchSF','EnclosedPorch',
                '3SsnPorch', 'ScreenPorch', 'PoolArea', 
                'PoolQC','Fence', 'MiscFeature', 'MiscVal', 
                'MoSold', 'YrSold', 'SaleType',
                'SaleCondition', 'SalePrice']

numeric_features = ["LotFrontage", "LotArea","OverallQual",
                   "OverallCond", "YearBuilt", "YearRemodAdd",
                   "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
                   "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
                   "LowQualFinSF", "GrLivArea", "BsmtFullBath",
                   "BsmtHalfBath", "FullBath", "HalfBath",
                   "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd",
                   "Fireplaces", "GarageCars", "GarageArea",
                   "WoodDeckSF", "OpenPorchSF", "EnclosedPorch",
                   "3SsnPorch", "ScreenPorch", "PoolArea",
                   "MiscVal", "MoSold", "YrSold","GarageYrBlt",
                   "MasVnrArea"]
categorical_features = ["MSZoning", "Street", "Alley", "LotShape",
                       "LandContour", "Utilities", "LotConfig",
                       "LandSlope", "Neighborhood", "Condition1",
                       "Condition2", "BldgType", "HouseStyle",
                       "RoofStyle", "RoofMatl", "Exterior1st",
                       "Exterior2nd", "ExterQual",
                       "ExterCond", "Foundation", "BsmtQual",
                       "BsmtCond", "BsmtExposure","BsmtFinType1",
                       "BsmtFinType2", "Heating", "HeatingQC",
                       "CentralAir", "Electrical","KitchenQual",
                       "Functional", "FireplaceQu", "GarageType",
                       "GarageFinish", "GarageQual",
                       "GarageCond", "PavedDrive", "PoolQC",
                       "Fence", "MiscFeature", "SaleType",
                       "SaleCondition"]

numeric_cleanup = pipeline.make_pipeline(
  impute.SimpleImputer(strategy="median"),
  preprocessing.StandardScaler())
categorical_cleanup = pipeline.make_pipeline(
  impute.SimpleImputer(strategy="constant", fill_value="NA"),
  preprocessing.OneHotEncoder(handle_unknown="ignore"))

cleanup = compose.make_column_transformer(
  (numeric_cleanup, numeric_features),
  (categorical_cleanup, categorical_features))
from sklearn import model_selection

cleanup.fit(data_train)
clean_train = cleanup.transform(data_train)
clean_test = cleanup.transform(data_test)

X_train, X_val, y_train, y_val = \
  model_selection.train_test_split(clean_train, data_train["SalePrice"].values)

X_train
import seaborn
seaborn.pairplot(data_train[[
    "SalePrice",#"LotFrontage", "LotArea","OverallQual",
    #"OverallCond", "YearBuilt", "YearRemodAdd",
    "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
    "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
    "LowQualFinSF"
]])
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_val, y_val))

y_pred = model.predict(clean_test)
submission = pd.DataFrame({
    "Id": data_test["Id"],
    "SalePrice": y_pred
})
submission.to_csv("submission.csv", index=False)
