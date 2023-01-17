# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv("../input/train.csv")

testDF = pd.read_csv("../input/test.csv")
cat_cols = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape", "LandContour",

           "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2",

           "BldgType", "HouseStyle", "OverallQual", "OverallCond", "RoofStyle", "RoofMatl",

           "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual", "ExterCond", "Foundation",

           "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating",

           "HeatingQC", "CentralAir", "Electrical","BsmtFullBath", "BsmtHalfBath","FullBath",

           "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional",

           "Fireplaces", "FireplaceQu", "GarageType", "GarageFinish", "GarageCars", "GarageQual",

           "GarageCond", "PavedDrive", "PoolQC", "Fence", "MiscFeature","MoSold", "SaleType",

           "SaleCondition"]

year_cols = ["YearBuilt", "YearRemodAdd", "GarageYrBlt","YrSold" ]

num_cols = ["LotFrontage", "LotArea", "MasVnrArea","BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF",

           "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", "GarageArea",

           "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",

           "MiscVal"]

y_col = ""
trainDF[num_cols].hist(figsize=[10,10])