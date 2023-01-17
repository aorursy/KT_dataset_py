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
import seaborn as sns



from matplotlib import pyplot as plt

from scipy import stats

from scipy.stats import norm, skew
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



print(f"Size of train dataset: {len(train)}")

print(f"Size of test dataset: {len(test)}")



df = train.copy()
train.head()
print(f"Number of columns: {len(df.columns)}")

print(f"Size of the train dataset: {len(df)}")
# Getting the columns that have null values and summing the amount of null values

df.loc[:, df.isnull().sum() > 0].isnull().sum()
df.corr().loc[:, "SalePrice"].abs().sort_values(ascending=False)
df.loc[:, "SalePrice"].hist(bins=40)
sns.distplot(df.loc[:, "SalePrice"] , fit=norm)
# df.loc[:, "SalePrice"] = np.log(df.loc[:, "SalePrice"])

# sns.distplot(df.loc[:, "SalePrice"], fit=norm)
sns.distplot(df.loc[:, "GrLivArea"], fit=norm)
df.loc[:, "GrLivArea"] = np.log(df.loc[:, "GrLivArea"])

test.loc[:, "GrLivArea"] = np.log(test.loc[:, "GrLivArea"])

sns.distplot(df.loc[:, "GrLivArea"], fit=norm)
sns.distplot(df.loc[:, "TotalBsmtSF"], fit=norm)
# Clipping outliers

# Only for training dataset

df = df.loc[df.loc[:, "TotalBsmtSF"] <= 6000, :]

sns.distplot(df.loc[:, "TotalBsmtSF"], fit=norm)
columns_to_encode = ["MSZoning", "Street", "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood", "Condition1",

                    "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",

                    "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC",

                    "CentralAir", "Electrical", "KitchenQual", "Functional", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",

                    "PavedDrive", "PoolQC", "Fence", "MiscFeature", "SaleType", "SaleCondition"]
MSZoning_mapping = {"A": 0, "C": 1, "FV": 2, "I": 3, "RH": 4,"RL": 5, "RP": 6, "RM": 7}

Street_mapping = {"Grvl": 0, "Pave": 1}

Alley_mapping = {"Grvl": 0, "Pave": 1, "NA": 2}

LotShape_mapping = {"Reg": 0, "IR1": 1, "IR2": 2, "IR3": 3}

LandContour_mapping = {"Lvl": 0, "Bnk": 1, "HLS": 2, "Low": 3}

Utilities_mapping = {"AllPub": 0, "NoSewr": 1, "NoSeWa": 2, "ELO": 3}

LotConfig_mapping = {"Inside": 0, "Corner": 1, "CulDSac": 2, "FR2": 3, "FR3": 4}

LandSlope_mapping = {"Gtl": 0, "Mod": 1, "Sev": 2}

Neighborhood_mapping = {"Blmngtn": 0, "Blueste": 1, "BrDale": 2, "BrkSide": 3, "ClearCr": 4, "CollgCr": 5, "Crawfor": 6, "Edwards": 7, 

                        "Gilbert": 8, "IDOTRR": 9, "MeadowV": 10, "Mitchel": 11, "Names": 12, "NoRidge": 13, "NPkVill": 14, "NridgHt": 15, 

                        "NWAmes": 16, "OldTown": 17, "SWISU": 18, "Sawyer": 19, "SawyerW": 20, "Somerst": 21, "StoneBr": 22, "Timber": 23, 

                        "Veenker": 24}

Condition1_mapping = {"Artery": 0, "Feedr": 1, "Norm": 2, "RRNn": 3, "RRAn": 4, "PosN": 5, "PosA": 6, "RRNe": 7, "RRAe": 8}

Condition2_mapping = {"Artery": 0, "Feedr": 1, "Norm": 2, "RRNn": 3, "RRAn": 4, "PosN": 5, "PosA": 6, "RRNe": 7, "RRAe": 8}

BldgType_mapping = {"1Fam": 0, "2FmCon": 1, "Duplx": 2, "TwnhsE": 3, "TwnhsI": 4}

HouseStyle_mapping = {"1Story": 0, "1.5Fin": 1, "1.5Unf": 2, "2Story": 3, "2.5Fin": 4, "2.5Unf": 5, "SFoyer": 6, "SLvl": 7}

RoofStyle_mapping = {"Flat": 0, "Gable": 1, "Gambrel": 2, "Hip": 3, "Mansard": 4, "Shed": 5}

RoofMatl_mapping = {"ClyTile": 0, "CompShg": 1, "Membran": 2, "Metal": 3, "Roll": 4, "Tar": 5, "WdShake": 5, "WdShngl": 6}

Exterior1st_mapping = {"AsbShng": 0, "AsphShn": 1, "BrkComm": 2, "BrkFace": 3, "CBlock": 4, "CemntBd": 5, "HdBoard": 6, "ImStucc": 7, "MetalSd": 8,

                       "Other": 9, "Plywood": 10, "PreCast": 11, "Stone": 12, "Stucco": 13, "VinylSd": 14, "Wd Sdng": 15, "WdShing": 16}

Exterior2nd_mapping = {"AsbShng": 0, "AsphShn": 1, "BrkComm": 2, "BrkFace": 3, "CBlock": 4, "CemntBd": 5, "HdBoard": 6, "ImStucc": 7, "MetalSd": 8,

                       "Other": 9, "Plywood": 10, "PreCast": 11, "Stone": 12, "Stucco": 13, "VinylSd": 14, "Wd Sdng": 15, "WdShing": 16}

MasVnrType_mapping = {"BrkCmn": 0, "BrkFace": 0, "CBlock": 0, "None": 0, "Stone": 0}

ExterQual_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}

ExterCond_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}

Foundation_mapping = {"BrkTil": 0, "CBlock": 1, "PConc": 2, "Slab": 3, "Stone": 4, "Wood": 5}

BsmtQual_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}

BsmtCond_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}

BsmtExposure_mapping = {"Gd": 0, "Av": 1, "Mn": 2, "No": 3, "NA": 4}

BsmtFinType1_mapping = {"GLQ": 0, "ALQ": 1, "BLQ": 2, "Rec": 3, "LwQ": 4, "Unf": 5, "NA": 6}

BsmtFinType2_mapping = {"GLQ": 0, "ALQ": 1, "BLQ": 2, "Rec": 3, "LwQ": 4, "Unf": 5, "NA": 6}

Heating_mapping = {"Floor": 0, "GasA": 1, "GasW": 2, "Grav": 3, "OthW": 4, "Wall": 5}

HeatingQC_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}

CentralAir_mapping = {"N": 0, "Y": 1}

Electrical_mapping = {"SBrkr": 0, "FuseA": 1, "FuseF": 2, "FuseP": 3, "Mix": 4}

KitchenQual_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4}

Functional_mapping = {"Typ": 0, "Min1": 1, "Min2": 2, "Mod": 3, "Maj1": 4, "Maj2": 5, "Sev": 6, "Sal": 7}

FireplaceQu_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}

GarageType_mapping = {"2Types": 0, "Attchd": 1, "Basment": 2, "BuiltIn": 3, "CarPort": 4, "Detchd": 5, "NA": 6}

GarageFinish_mapping = {"Fin": 0, "RFn": 1, "Unf": 2, "NA": 3}

GarageQual_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}

GarageCond_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}

PavedDrive_mapping = {"Y": 0, "P": 1, "N": 2}

PoolQC_mapping = {"Ex": 0, "Gd": 1, "TA": 2, "Fa": 3, "Po": 4, "NA": 5}

Fence_mapping = {"GdPrv": 0, "MnPrv": 1, "GdWo": 2, "MnWw": 3, "NA": 4}

MiscFeature_mapping = {"Elev": 0, "Gar2": 1, "Othr": 2, "Shed": 3, "TenC": 4, "NA": 5}

SaleType_mapping = {"WD": 0, "CWD": 1, "VWD": 2, "New": 3, "COD": 4, "Con": 5, "ConLw": 6, "ConLI": 7, "ConLD": 8, "Oth": 9}

SaleCondition_mapping = {"Normal": 0, "Abnorml": 1, "AdjLand": 2, "Alloca": 3, "Family": 4, "Partial": 5}
for column in columns_to_encode:

    mapping = eval(f"{column}_mapping")

    df.loc[:, column] = df.loc[:, column].map(mapping)

    test.loc[:, column] = test.loc[:, column].map(mapping)
# The classes with missing values, that represents non exitence of this features, we will fill the representing class

columns = ["Alley", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for column in columns:

    # In all the dicts, tha na value is the last key

    mapping = eval(f"{column}_mapping")

    na_key = list(mapping.keys())[-1]

    

    df.loc[df.loc[:, column].isnull(), column] = mapping[na_key]

    test.loc[test.loc[:, column].isnull(), column] = mapping[na_key]
# Attempt to fill with the median based on the column with the highest correlation

# We run it twice, because the order of the features being filled can be not optimal

# WE CANNOT FILL THE TARGET COLUMN

for i in range(2):

    missing_values_columns = df.loc[:, df.isnull().sum() > 0].isnull().sum().sort_values().index.tolist()



    for column in missing_values_columns:

        if column != "SalePrice":

            most_correlated_column = df.corr().loc[:, column].abs().sort_values(ascending=False).index.tolist()[1]

            if most_correlated_column == "SalePrice":

                most_correlated_column = df.corr().loc[:, column].abs().sort_values(ascending=False).index.tolist()[2]

            

            df.loc[df.loc[:, column].isnull(), column] = df.groupby(most_correlated_column)[column].transform("median")            

            test.loc[test.loc[:, column].isnull(), column] = test.groupby(most_correlated_column)[column].transform("median")
df.loc[:, df.isnull().sum() > 0].isnull().sum().sort_values()
# For most columns, we can just fill with the most common value

for column in ["Exterior2nd", "RoofMatl", "LotFrontage", "Exterior1st", "GarageCars", "GarageArea", "MSZoning"]:

    df.loc[df.loc[:, column].isnull(), column] = df.loc[:, column].mode().iloc[0]

    test.loc[test.loc[:, column].isnull(), column] = test.loc[:, column].mode().iloc[0]
# BsmtFinType2: Rating of basement finished area (if multiple type)

# As this column has only one missing value, we'll fill with the mode for the values greater than 0

df.loc[df.loc[:, "BsmtFinType2"].isnull(), "BsmtFinType2"] = df.loc[df.loc[:, "BsmtFinSF2"] > 0, "BsmtFinSF2"].mode().iloc[0]

test.loc[test.loc[:, "BsmtFinType2"].isnull(), "BsmtFinType2"] = test.loc[test.loc[:, "BsmtFinSF2"] > 0, "BsmtFinSF2"].mode().iloc[0]
# GarageYrBlt: Year garage was built

# As 6 represents that the house has no garage, we'll fill these values with 0

df.loc[df.loc[:, "GarageYrBlt"].isnull(), ["GarageYrBlt", "GarageType"]]
df.loc[df.loc[:, "GarageYrBlt"].isnull(), "GarageYrBlt"] = 0

test.loc[test.loc[:, "GarageYrBlt"].isnull(), "GarageYrBlt"] = 0
# MasVnrArea: Masonry veneer area in square feet

# If there's no mansonry, we fill with 0

df.loc[df.loc[:, "MasVnrArea"].isnull()] = 0

test.loc[test.loc[:, "MasVnrArea"].isnull()] = 0
# As the most correlated column didn't fill the null values, we'll use the second most correlated column

df.loc[df.loc[:, "BldgType"].isnull(), "BldgType"] = df.groupby("LotFrontage")["BldgType"].transform("median")

test.loc[test.loc[:, "BldgType"].isnull(), "BldgType"] = test.groupby("LotFrontage")["BldgType"].transform("median")



# If any row is still with null values, we'll fill with the mode

df.loc[df.loc[:, "BldgType"].isnull(), "BldgType"] = df.loc[:, "BldgType"].mode().iloc[0]

test.loc[test.loc[:, "BldgType"].isnull(), "BldgType"] = test.loc[:, "BldgType"].mode().iloc[0]
df.loc[:, df.isnull().sum() > 0].isnull().sum().sort_values()
import xgboost as xgb



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 

            'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 

            'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 

            'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 

            'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 

            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional',

            'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 

            'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 

            'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

target = "SalePrice"
train, dev = train_test_split(df, test_size=0.2, random_state=4)
D_train = xgb.DMatrix(train.loc[:, features], train.loc[:, target])

D_dev = xgb.DMatrix(dev.loc[:, features])



D_test = xgb.DMatrix(test.loc[:, features])
param = { 

    "objective": "reg:squarederror",  

}



model = xgb.train(param, D_train)

predictions = model.predict(D_dev)



error = mean_squared_error(dev.loc[:, target], predictions)



print(f"Initial error: {error}")
D_total = xgb.DMatrix(df.loc[:, features], df.loc[:, target])



param = { 

    "objective": "reg:squarederror",  

}



model = xgb.train(param, D_total)
predictions = model.predict(D_test)
test.loc[:, "SalePrice"] = predictions
test.head()
submission = test.loc[:, ["Id", "SalePrice"]]

submission.to_csv("/kaggle/working/submission.csv", index=False)