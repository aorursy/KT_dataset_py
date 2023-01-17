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
import seaborn as sns

from scipy import stats

from sklearn.model_selection import KFold, train_test_split

from sklearn.preprocessing import StandardScaler
import keras

from keras.engine.input_layer import Input

from keras.layers import Dense, Activation, Dropout, Add, PReLU, LeakyReLU

from keras.layers.normalization import BatchNormalization

from keras.optimizers import SGD, Adadelta, Adagrad, Adam, RMSprop, Adamax, Nadam

from keras.models import Model

from keras.regularizers import l1, l2, l1_l2

from keras.callbacks import EarlyStopping
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

df_train.head()
df_train.describe()
df_train_pp = pd.DataFrame(df_train["Id"], columns = ["Id"])

df_test_pp = pd.DataFrame(df_test["Id"], columns = ["Id"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "MSSubClass"]], columns = ["MSSubClass"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "MSSubClass"]], columns = ["MSSubClass"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["MSSubClass"]], columns = ["MSSubClass"])[["MSSubClass_30", "MSSubClass_60"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["MSSubClass"]], columns = ["MSSubClass"])[["MSSubClass_30", "MSSubClass_60"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "MSZoning"]], columns = ["MSZoning"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "MSZoning"]], columns = ["MSZoning"]).corr()["SalePrice"])
sns.heatmap(pd.concat([

    df_train[["SalePrice"]],

    pd.get_dummies(df_train[["MSZoning"]], columns = ["MSZoning"]) + np.expand_dims(pd.get_dummies(df_train[["MSZoning"]], columns = ["MSZoning"])["MSZoning_C (all)"].values, axis = -1),

], axis = 1).corr())

print(pd.concat([

    df_train[["SalePrice"]],

    pd.get_dummies(df_train[["MSZoning"]], columns = ["MSZoning"]) + np.expand_dims(pd.get_dummies(df_train[["MSZoning"]], columns = ["MSZoning"])["MSZoning_C (all)"].values, axis = -1),

], axis = 1).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, (pd.get_dummies(df_train[["MSZoning"]], columns = ["MSZoning"]) + np.expand_dims(pd.get_dummies(df_train[["MSZoning"]], columns = ["MSZoning"])["MSZoning_C (all)"].values, axis = -1))[["MSZoning_RL", "MSZoning_RM"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, (pd.get_dummies(df_test[["MSZoning"]], columns = ["MSZoning"]) + np.expand_dims(pd.get_dummies(df_test[["MSZoning"]], columns = ["MSZoning"])["MSZoning_C (all)"].values, axis = -1))[["MSZoning_RL", "MSZoning_RM"]]], axis = 1)
sns.heatmap(df_train[["SalePrice", "LotFrontage", "LotArea"]].corr())

print(df_train[["SalePrice", "LotFrontage", "LotArea"]].corr()["SalePrice"])
'''

df_temp = df_train["LotFrontage"].fillna(df_train["LotFrontage"].mean())

def convert_LotFrontage(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["LotFrontage"].fillna(df_train["LotFrontage"].mean()).map(convert_LotFrontage)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["LotFrontage"].fillna(df_test["LotFrontage"].mean()).map(convert_LotFrontage)], axis = 1)

'''
print(np.expand_dims(df_train["LotFrontage"].fillna(df_train["LotFrontage"].mean()).values, axis = 1))
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["LotFrontage"].fillna(df_train["LotFrontage"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["LotFrontage"].fillna(df_train["LotFrontage"].mean()).values, axis = 1)), columns = ["LotFrontage"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["LotFrontage"].fillna(df_test["LotFrontage"].mean()).values, axis = 1)), columns = ["LotFrontage"])], axis = 1)
'''

df_temp = df_train["LotArea"].fillna(df_train["LotArea"].mean())

def convert_LotArea(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["LotArea"].fillna(df_train["LotArea"].mean()).map(convert_LotArea)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["LotArea"].fillna(df_test["LotArea"].mean()).map(convert_LotArea)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["LotArea"].fillna(df_train["LotArea"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["LotArea"].fillna(df_train["LotArea"].mean()).values, axis = 1)), columns = ["LotArea"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["LotArea"].fillna(df_test["LotArea"].mean()).values, axis = 1)), columns = ["LotArea"])], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Street", "Alley"]], columns = ["Street", "Alley"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Street", "Alley"]], columns = ["Street", "Alley"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "LotShape"]], columns = ["LotShape"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "LotShape"]], columns = ["LotShape"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["LotShape"]], columns = ["LotShape"])[["LotShape_IR1", "LotShape_Reg"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["LotShape"]], columns = ["LotShape"])[["LotShape_IR1", "LotShape_Reg"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "LandContour"]], columns = ["LandContour"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "LandContour"]], columns = ["LandContour"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Utilities"]], columns = ["Utilities"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Utilities"]], columns = ["Utilities"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "LotConfig"]], columns = ["LotConfig"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "LotConfig"]], columns = ["LotConfig"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "LandSlope"]], columns = ["LandSlope"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "LandSlope"]], columns = ["LandSlope"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Neighborhood"]], columns = ["Neighborhood"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Neighborhood"]], columns = ["Neighborhood"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["Neighborhood"]], columns = ["Neighborhood"])[["Neighborhood_NoRidge", "Neighborhood_NridgHt", "Neighborhood_StoneBr"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["Neighborhood"]], columns = ["Neighborhood"])[["Neighborhood_NoRidge", "Neighborhood_NridgHt", "Neighborhood_StoneBr"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Condition1", "Condition2"]], columns = ["Condition1", "Condition2"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Condition1", "Condition2"]], columns = ["Condition1", "Condition2"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "BldgType"]], columns = ["BldgType"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "BldgType"]], columns = ["BldgType"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "HouseStyle"]], columns = ["HouseStyle"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "HouseStyle"]], columns = ["HouseStyle"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["HouseStyle"]], columns = ["HouseStyle"])[["HouseStyle_2Story"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["HouseStyle"]], columns = ["HouseStyle"])[["HouseStyle_2Story"]]], axis = 1)
sns.heatmap(df_train[["SalePrice", "OverallQual", "OverallCond"]].corr())

print(df_train[["SalePrice", "OverallQual", "OverallCond"]].corr()["SalePrice"])
'''

df_temp = df_train["OverallQual"].fillna(df_train["OverallQual"].mean())

def convert_OverallQual(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["OverallQual"].fillna(df_train["OverallQual"].mean()).map(convert_OverallQual)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["OverallQual"].fillna(df_test["OverallQual"].mean()).map(convert_OverallQual)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["OverallQual"].fillna(df_train["OverallQual"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["OverallQual"].fillna(df_train["OverallQual"].mean()).values, axis = 1)), columns = ["OverallQual"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["OverallQual"].fillna(df_test["OverallQual"].mean()).values, axis = 1)), columns = ["OverallQual"])], axis = 1)
sns.heatmap(df_train[["SalePrice", "YearBuilt", "YearRemodAdd"]].corr())

print(df_train[["SalePrice", "YearBuilt", "YearRemodAdd"]].corr()["SalePrice"])
'''

df_temp = df_train["YearBuilt"].fillna(df_train["YearBuilt"].mean())

def convert_YearBuilt(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["YearBuilt"].fillna(df_train["YearBuilt"].mean()).map(convert_YearBuilt)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["YearBuilt"].fillna(df_test["YearBuilt"].mean()).map(convert_YearBuilt)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["YearBuilt"].fillna(df_train["YearBuilt"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["YearBuilt"].fillna(df_train["YearBuilt"].mean()).values, axis = 1)), columns = ["YearBuilt"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["YearBuilt"].fillna(df_test["YearBuilt"].mean()).values, axis = 1)), columns = ["YearBuilt"])], axis = 1)
'''

df_temp = df_train["YearRemodAdd"].fillna(df_train["YearRemodAdd"].mean())

def convert_YearRemodAdd(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["YearRemodAdd"].fillna(df_train["YearRemodAdd"].mean()).map(convert_YearRemodAdd)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["YearRemodAdd"].fillna(df_test["YearRemodAdd"].mean()).map(convert_YearRemodAdd)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["YearRemodAdd"].fillna(df_train["YearRemodAdd"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["YearRemodAdd"].fillna(df_train["YearRemodAdd"].mean()).values, axis = 1)), columns = ["YearRemodAdd"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["YearRemodAdd"].fillna(df_test["YearRemodAdd"].mean()).values, axis = 1)), columns = ["YearRemodAdd"])], axis = 1)
sns.heatmap(pd.concat([df_train[["SalePrice"]], df_train[["YearBuilt", "YearRemodAdd"]].max(axis = 1)], axis = 1).corr())

print(pd.concat([df_train[["SalePrice"]], df_train[["YearBuilt", "YearRemodAdd"]].max(axis = 1)], axis = 1).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "RoofStyle", "RoofMatl"]], columns = ["RoofStyle", "RoofMatl"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "RoofStyle", "RoofMatl"]], columns = ["RoofStyle", "RoofMatl"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["RoofStyle"]], columns = ["RoofStyle"])[["RoofStyle_Gable", "RoofStyle_Hip"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["RoofStyle"]], columns = ["RoofStyle"])[["RoofStyle_Gable", "RoofStyle_Hip"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Exterior1st", "Exterior2nd"]], columns = ["Exterior1st", "Exterior2nd"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Exterior1st", "Exterior2nd"]], columns = ["Exterior1st", "Exterior2nd"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["Exterior1st", "Exterior2nd"]], columns = ["Exterior1st", "Exterior2nd"])[["Exterior1st_VinylSd", "Exterior2nd_VinylSd"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["Exterior1st", "Exterior2nd"]], columns = ["Exterior1st", "Exterior2nd"])[["Exterior1st_VinylSd", "Exterior2nd_VinylSd"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "MasVnrType", "MasVnrArea"]], columns = ["MasVnrType"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "MasVnrType", "MasVnrArea"]], columns = ["MasVnrType"]).corr()["SalePrice"])
'''

df_temp = df_train["MasVnrArea"].fillna(df_train["MasVnrArea"].mean())

def convert_MasVnrArea(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["MasVnrArea"].fillna(df_train["MasVnrArea"].mean()).map(convert_MasVnrArea)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["MasVnrArea"].fillna(df_test["MasVnrArea"].mean()).map(convert_MasVnrArea)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["MasVnrArea"].fillna(df_train["MasVnrArea"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["MasVnrArea"].fillna(df_train["MasVnrArea"].mean()).values, axis = 1)), columns = ["MasVnrArea"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["MasVnrArea"].fillna(df_test["MasVnrArea"].mean()).values, axis = 1)), columns = ["MasVnrArea"])], axis = 1)
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["MasVnrType"]], columns = ["MasVnrType"])[["MasVnrType_None", "MasVnrType_Stone"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["MasVnrType"]], columns = ["MasVnrType"])[["MasVnrType_None", "MasVnrType_Stone"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "ExterQual", "ExterCond"]], columns = ["ExterQual", "ExterCond"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "ExterQual", "ExterCond"]], columns = ["ExterQual", "ExterCond"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["ExterQual"]], columns = ["ExterQual"])[["ExterQual_Ex", "ExterQual_Gd", "ExterQual_TA"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["ExterQual"]], columns = ["ExterQual"])[["ExterQual_Ex", "ExterQual_Gd", "ExterQual_TA"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Foundation"]], columns = ["Foundation"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Foundation"]], columns = ["Foundation"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["Foundation"]], columns = ["Foundation"])[["Foundation_BrkTil", "Foundation_CBlock", "Foundation_PConc"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["Foundation"]], columns = ["Foundation"])[["Foundation_BrkTil", "Foundation_CBlock", "Foundation_PConc"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "BsmtQual", "BsmtCond", "BsmtExposure"]], columns = ["BsmtQual", "BsmtCond", "BsmtExposure"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "BsmtQual", "BsmtCond", "BsmtExposure"]], columns = ["BsmtQual", "BsmtCond", "BsmtExposure"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["BsmtQual", "BsmtExposure"]], columns = ["BsmtQual", "BsmtExposure"])[["BsmtQual_Ex", "BsmtQual_Gd", "BsmtQual_TA", "BsmtExposure_Gd", "BsmtExposure_No"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["BsmtQual", "BsmtExposure"]], columns = ["BsmtQual", "BsmtExposure"])[["BsmtQual_Ex", "BsmtQual_Gd", "BsmtQual_TA", "BsmtExposure_Gd", "BsmtExposure_No"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]], columns = ["BsmtFinType1", "BsmtFinType2"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF"]], columns = ["BsmtFinType1", "BsmtFinType2"]).corr()["SalePrice"])
'''

df_temp = df_train["BsmtFinSF1"].fillna(df_train["BsmtFinSF1"].mean())

def convert_BsmtFinSF1(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["BsmtFinSF1"].fillna(df_train["BsmtFinSF1"].mean()).map(convert_BsmtFinSF1)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["BsmtFinSF1"].fillna(df_test["BsmtFinSF1"].mean()).map(convert_BsmtFinSF1)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["BsmtFinSF1"].fillna(df_train["BsmtFinSF1"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["BsmtFinSF1"].fillna(df_train["BsmtFinSF1"].mean()).values, axis = 1)), columns = ["BsmtFinSF1"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["BsmtFinSF1"].fillna(df_test["BsmtFinSF1"].mean()).values, axis = 1)), columns = ["BsmtFinSF1"])], axis = 1)
'''

df_temp = df_train["BsmtUnfSF"].fillna(df_train["BsmtUnfSF"].mean())

def convert_BsmtUnfSF(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["BsmtUnfSF"].fillna(df_train["BsmtUnfSF"].mean()).map(convert_BsmtUnfSF)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["BsmtUnfSF"].fillna(df_test["BsmtUnfSF"].mean()).map(convert_BsmtUnfSF)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["BsmtUnfSF"].fillna(df_train["BsmtUnfSF"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["BsmtUnfSF"].fillna(df_train["BsmtUnfSF"].mean()).values, axis = 1)), columns = ["BsmtUnfSF"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["BsmtUnfSF"].fillna(df_test["BsmtUnfSF"].mean()).values, axis = 1)), columns = ["BsmtUnfSF"])], axis = 1)
'''

df_temp = df_train["TotalBsmtSF"].fillna(df_train["TotalBsmtSF"].mean())

def convert_TotalBsmtSF(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["TotalBsmtSF"].fillna(df_train["TotalBsmtSF"].mean()).map(convert_TotalBsmtSF)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["TotalBsmtSF"].fillna(df_test["TotalBsmtSF"].mean()).map(convert_TotalBsmtSF)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["TotalBsmtSF"].fillna(df_train["TotalBsmtSF"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["TotalBsmtSF"].fillna(df_train["TotalBsmtSF"].mean()).values, axis = 1)), columns = ["TotalBsmtSF"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["TotalBsmtSF"].fillna(df_test["TotalBsmtSF"].mean()).values, axis = 1)), columns = ["TotalBsmtSF"])], axis = 1)
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["BsmtFinType1"]], columns = ["BsmtFinType1"])[["BsmtFinType1_GLQ"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["BsmtFinType1"]], columns = ["BsmtFinType1"])[["BsmtFinType1_GLQ"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Heating", "HeatingQC"]], columns = ["Heating", "HeatingQC"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Heating", "HeatingQC"]], columns = ["Heating", "HeatingQC"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["HeatingQC"]], columns = ["HeatingQC"])[["HeatingQC_Ex", "HeatingQC_TA"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["HeatingQC"]], columns = ["HeatingQC"])[["HeatingQC_Ex", "HeatingQC_TA"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "CentralAir"]], columns = ["CentralAir"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "CentralAir"]], columns = ["CentralAir"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["CentralAir"]], columns = ["CentralAir"])[["CentralAir_Y"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["CentralAir"]], columns = ["CentralAir"])[["CentralAir_Y"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Electrical"]], columns = ["Electrical"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Electrical"]], columns = ["Electrical"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["Electrical"]], columns = ["Electrical"])[["Electrical_SBrkr"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["Electrical"]], columns = ["Electrical"])[["Electrical_SBrkr"]]], axis = 1)
sns.heatmap(df_train[["SalePrice", "1stFlrSF", "2ndFlrSF"]].corr())

print(df_train[["SalePrice", "1stFlrSF", "2ndFlrSF"]].corr()["SalePrice"])
'''

df_temp = df_train["1stFlrSF"].fillna(df_train["1stFlrSF"].mean())

def convert_1stFlrSF(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["1stFlrSF"].fillna(df_train["1stFlrSF"].mean()).map(convert_1stFlrSF)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["1stFlrSF"].fillna(df_test["1stFlrSF"].mean()).map(convert_1stFlrSF)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["1stFlrSF"].fillna(df_train["1stFlrSF"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["1stFlrSF"].fillna(df_train["1stFlrSF"].mean()).values, axis = 1)), columns = ["1stFlrSF"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["1stFlrSF"].fillna(df_test["1stFlrSF"].mean()).values, axis = 1)), columns = ["1stFlrSF"])], axis = 1)
'''

df_temp = df_train["2ndFlrSF"].fillna(df_train["2ndFlrSF"].mean())

def convert_2ndFlrSF(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["2ndFlrSF"].fillna(df_train["2ndFlrSF"].mean()).map(convert_2ndFlrSF)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["2ndFlrSF"].fillna(df_test["2ndFlrSF"].mean()).map(convert_2ndFlrSF)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["2ndFlrSF"].fillna(df_train["2ndFlrSF"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["2ndFlrSF"].fillna(df_train["2ndFlrSF"].mean()).values, axis = 1)), columns = ["2ndFlrSF"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["2ndFlrSF"].fillna(df_test["2ndFlrSF"].mean()).values, axis = 1)), columns = ["2ndFlrSF"])], axis = 1)
sns.heatmap(df_train[["SalePrice", "LowQualFinSF"]].corr())

print(df_train[["SalePrice", "LowQualFinSF"]].corr()["SalePrice"])
sns.heatmap(df_train[["SalePrice", "GrLivArea"]].corr())

print(df_train[["SalePrice", "GrLivArea"]].corr()["SalePrice"])
'''

df_temp = df_train["GrLivArea"].fillna(df_train["GrLivArea"].mean())

def convert_GrLivArea(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["GrLivArea"].fillna(df_train["GrLivArea"].mean()).map(convert_GrLivArea)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["GrLivArea"].fillna(df_test["GrLivArea"].mean()).map(convert_GrLivArea)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["GrLivArea"].fillna(df_train["GrLivArea"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["GrLivArea"].fillna(df_train["GrLivArea"].mean()).values, axis = 1)), columns = ["GrLivArea"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["GrLivArea"].fillna(df_test["GrLivArea"].mean()).values, axis = 1)), columns = ["GrLivArea"])], axis = 1)
sns.heatmap(df_train[["SalePrice", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]].corr())

print(df_train[["SalePrice", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]].corr()["SalePrice"])
'''

df_temp = df_train["BsmtFullBath"].fillna(df_train["BsmtFullBath"].mean())

def convert_BsmtFullBath(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["BsmtFullBath"].fillna(df_train["BsmtFullBath"].mean()).map(convert_BsmtFullBath)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["BsmtFullBath"].fillna(df_test["BsmtFullBath"].mean()).map(convert_BsmtFullBath)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["BsmtFullBath"].fillna(df_train["BsmtFullBath"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["BsmtFullBath"].fillna(df_train["BsmtFullBath"].mean()).values, axis = 1)), columns = ["BsmtFullBath"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["BsmtFullBath"].fillna(df_test["BsmtFullBath"].mean()).values, axis = 1)), columns = ["BsmtFullBath"])], axis = 1)
'''

df_temp = df_train["FullBath"].fillna(df_train["FullBath"].mean())

def convert_FullBath(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["FullBath"].fillna(df_train["FullBath"].mean()).map(convert_FullBath)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["FullBath"].fillna(df_test["FullBath"].mean()).map(convert_FullBath)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["FullBath"].fillna(df_train["FullBath"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["FullBath"].fillna(df_train["FullBath"].mean()).values, axis = 1)), columns = ["FullBath"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["FullBath"].fillna(df_test["FullBath"].mean()).values, axis = 1)), columns = ["FullBath"])], axis = 1)
'''

df_temp = df_train["HalfBath"].fillna(df_train["HalfBath"].mean())

def convert_HalfBath(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["HalfBath"].fillna(df_train["HalfBath"].mean()).map(convert_HalfBath)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["HalfBath"].fillna(df_test["HalfBath"].mean()).map(convert_HalfBath)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["HalfBath"].fillna(df_train["HalfBath"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["HalfBath"].fillna(df_train["HalfBath"].mean()).values, axis = 1)), columns = ["HalfBath"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["HalfBath"].fillna(df_test["HalfBath"].mean()).values, axis = 1)), columns = ["HalfBath"])], axis = 1)
sns.heatmap(df_train[["SalePrice", "BedroomAbvGr"]].corr())

print(df_train[["SalePrice", "BedroomAbvGr"]].corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "KitchenAbvGr", "KitchenQual"]], columns = ["KitchenQual"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "KitchenAbvGr", "KitchenQual"]], columns = ["KitchenQual"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["KitchenQual"]], columns = ["KitchenQual"])[["KitchenQual_Ex", "KitchenQual_Gd", "KitchenQual_TA"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["KitchenQual"]], columns = ["KitchenQual"])[["KitchenQual_Ex", "KitchenQual_Gd", "KitchenQual_TA"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Functional"]], columns = ["Functional"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Functional"]], columns = ["Functional"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Fireplaces", "FireplaceQu"]], columns = ["FireplaceQu"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Fireplaces", "FireplaceQu"]], columns = ["FireplaceQu"]).corr()["SalePrice"])
'''

df_temp = df_train["Fireplaces"].fillna(df_train["Fireplaces"].mean())

def convert_Fireplaces(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["Fireplaces"].fillna(df_train["Fireplaces"].mean()).map(convert_Fireplaces)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["Fireplaces"].fillna(df_test["Fireplaces"].mean()).map(convert_Fireplaces)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["Fireplaces"].fillna(df_train["Fireplaces"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["Fireplaces"].fillna(df_train["Fireplaces"].mean()).values, axis = 1)), columns = ["Fireplaces"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["Fireplaces"].fillna(df_test["Fireplaces"].mean()).values, axis = 1)), columns = ["Fireplaces"])], axis = 1)
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["FireplaceQu"]], columns = ["FireplaceQu"])[["FireplaceQu_Ex", "FireplaceQu_Gd"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["FireplaceQu"]], columns = ["FireplaceQu"])[["FireplaceQu_Ex", "FireplaceQu_Gd"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"]], columns = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"]], columns = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]).corr()["SalePrice"])
'''

df_temp = df_train["GarageYrBlt"].fillna(df_train["GarageYrBlt"].mean())

def convert_GarageYrBlt(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["GarageYrBlt"].fillna(df_train["GarageYrBlt"].mean()).map(convert_GarageYrBlt)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["GarageYrBlt"].fillna(df_test["GarageYrBlt"].mean()).map(convert_GarageYrBlt)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["GarageYrBlt"].fillna(df_train["GarageYrBlt"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["GarageYrBlt"].fillna(df_train["GarageYrBlt"].mean()).values, axis = 1)), columns = ["GarageYrBlt"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["GarageYrBlt"].fillna(df_test["GarageYrBlt"].mean()).values, axis = 1)), columns = ["GarageYrBlt"])], axis = 1)
'''

df_temp = df_train["GarageCars"].fillna(df_train["GarageCars"].mean())

def convert_GarageCars(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["GarageCars"].fillna(df_train["GarageCars"].mean()).map(convert_GarageCars)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["GarageCars"].fillna(df_test["GarageCars"].mean()).map(convert_GarageCars)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["GarageCars"].fillna(df_train["GarageCars"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["GarageCars"].fillna(df_train["GarageCars"].mean()).values, axis = 1)), columns = ["GarageCars"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["GarageCars"].fillna(df_test["GarageCars"].mean()).values, axis = 1)), columns = ["GarageCars"])], axis = 1)
'''

df_temp = df_train["GarageArea"].fillna(df_train["GarageArea"].mean())

def convert_GarageArea(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["GarageArea"].fillna(df_train["GarageArea"].mean()).map(convert_GarageArea)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["GarageArea"].fillna(df_test["GarageArea"].mean()).map(convert_GarageArea)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["GarageArea"].fillna(df_train["GarageArea"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["GarageArea"].fillna(df_train["GarageArea"].mean()).values, axis = 1)), columns = ["GarageArea"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["GarageArea"].fillna(df_test["GarageArea"].mean()).values, axis = 1)), columns = ["GarageArea"])], axis = 1)
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["GarageType", "GarageFinish", "GarageQual", "GarageCond"]], columns = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"])[["GarageType_Attchd", "GarageType_BuiltIn", "GarageType_Detchd", "GarageFinish_Fin", "GarageFinish_Unf", "GarageQual_TA", "GarageCond_TA"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["GarageType", "GarageFinish", "GarageQual", "GarageCond"]], columns = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"])[["GarageType_Attchd", "GarageType_BuiltIn", "GarageType_Detchd", "GarageFinish_Fin", "GarageFinish_Unf", "GarageQual_TA", "GarageCond_TA"]]], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "PavedDrive"]], columns = ["PavedDrive"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "PavedDrive"]], columns = ["PavedDrive"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["PavedDrive"]], columns = ["PavedDrive"])[["PavedDrive_N", "PavedDrive_Y"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["PavedDrive"]], columns = ["PavedDrive"])[["PavedDrive_N", "PavedDrive_Y"]]], axis = 1)
sns.heatmap(df_train[["SalePrice", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].corr())

print(df_train[["SalePrice", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"]].corr()["SalePrice"])
'''

df_temp = df_train["WoodDeckSF"].fillna(df_train["WoodDeckSF"].mean())

def convert_WoodDeckSF(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["WoodDeckSF"].fillna(df_train["WoodDeckSF"].mean()).map(convert_WoodDeckSF)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["WoodDeckSF"].fillna(df_test["WoodDeckSF"].mean()).map(convert_WoodDeckSF)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["WoodDeckSF"].fillna(df_train["WoodDeckSF"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["WoodDeckSF"].fillna(df_train["WoodDeckSF"].mean()).values, axis = 1)), columns = ["WoodDeckSF"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["WoodDeckSF"].fillna(df_test["WoodDeckSF"].mean()).values, axis = 1)), columns = ["WoodDeckSF"])], axis = 1)
'''

df_temp = df_train["OpenPorchSF"].fillna(df_train["OpenPorchSF"].mean())

def convert_OpenPorchSF(value, temp = df_temp):

    minValue = temp.min()

    maxValue = temp.max()

    preprocessed = (value - minValue) / (maxValue - minValue)

    return preprocessed



df_train_pp = pd.concat([df_train_pp, df_train["OpenPorchSF"].fillna(df_train["OpenPorchSF"].mean()).map(convert_OpenPorchSF)], axis = 1)

df_test_pp = pd.concat([df_test_pp, df_test["OpenPorchSF"].fillna(df_test["OpenPorchSF"].mean()).map(convert_OpenPorchSF)], axis = 1)

'''
scaler = StandardScaler()

scaler.fit(np.expand_dims(df_train["OpenPorchSF"].fillna(df_train["OpenPorchSF"].mean()).values, axis = 1))

df_train_pp = pd.concat([df_train_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_train["OpenPorchSF"].fillna(df_train["OpenPorchSF"].mean()).values, axis = 1)), columns = ["OpenPorchSF"])], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.DataFrame(scaler.transform(np.expand_dims(df_test["OpenPorchSF"].fillna(df_test["OpenPorchSF"].mean()).values, axis = 1)), columns = ["OpenPorchSF"])], axis = 1)
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "PoolArea", "PoolQC"]], columns = ["PoolQC"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "PoolArea", "PoolQC"]], columns = ["PoolQC"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "Fence"]], columns = ["Fence"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "Fence"]], columns = ["Fence"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "MiscFeature", "MiscVal"]], columns = ["MiscFeature"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "MiscFeature", "MiscVal"]], columns = ["MiscFeature"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "MoSold", "YrSold"]], columns = ["MoSold"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "MoSold", "YrSold"]], columns = ["MoSold"]).corr()["SalePrice"])
sns.heatmap(pd.get_dummies(df_train[["SalePrice", "SaleType", "SaleCondition"]], columns = ["SaleType", "SaleCondition"]).corr())

print(pd.get_dummies(df_train[["SalePrice", "SaleType", "SaleCondition"]], columns = ["SaleType", "SaleCondition"]).corr()["SalePrice"])
df_train_pp = pd.concat([df_train_pp, pd.get_dummies(df_train[["SaleType", "SaleCondition"]], columns = ["SaleType", "SaleCondition"])[["SaleType_New", "SaleType_WD", "SaleCondition_Partial"]]], axis = 1)

df_test_pp = pd.concat([df_test_pp, pd.get_dummies(df_test[["SaleType", "SaleCondition"]], columns = ["SaleType", "SaleCondition"])[["SaleType_New", "SaleType_WD", "SaleCondition_Partial"]]], axis = 1)
df_train_pp.head()
df_train_pp = df_train_pp.drop(["Id"], axis = 1)

df_test_pp = df_test_pp.drop(["Id"], axis = 1)
df_train_prices = df_train["SalePrice"]
na_train_x = df_train_pp.values

na_train_y = df_train_prices.values

na_test_x = df_test_pp.values

na_test_id = np.expand_dims(df_test["Id"].values, axis = -1)
X_train, X_test, Y_train, Y_test = train_test_split(na_train_x, na_train_y, test_size = 0.1)
input_units = na_train_x.shape[1]
def build_model(input_units = input_units):

    input_layer = Input((input_units,))

    

    x = Dense(units = 512, activity_regularizer=l1_l2())(input_layer)

    x = LeakyReLU()(x)

    #x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    

    x = Dense(units = 256, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    #x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    

    x = Dense(units = 128, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    #x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    

    x = Dense(units = 64, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    #x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    

    x = Dense(units = 32, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    #x = BatchNormalization()(x)

    x = Dropout(0.25)(x)

    

    x = Dense(units = 16, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    x = Dropout(0.2)(x)

    

    x = Dense(units = 8, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    x = Dropout(0.15)(x)

    

    x = Dense(units = 4, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    

    x = Dense(units = 2, activity_regularizer=l1_l2())(x)

    x = LeakyReLU()(x)

    

    x = Dense(units = 1)(x)

    

    model = Model(inputs = input_layer, outputs = x)

    model.compile(optimizer = Nadam(), loss = 'mean_squared_error', metrics = ['mean_absolute_error'])

    return model
kf = KFold(n_splits = 10, shuffle = True)

all_loss = []

all_val_loss = []

all_mae = []

all_val_mae = []

epochs = 10000
'''

for train_index, val_index in kf.split(X_train, Y_train):

    train_data = X_train[train_index]

    train_label = Y_train[train_index]

    val_data = X_train[val_index]

    val_label = Y_train[val_index]

    

    model = build_model()

    earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience=100, verbose=1, mode='auto')

    history = model.fit(x = train_data, y = train_label, epochs = epochs, batch_size = 200, validation_data = (val_data, val_label), callbacks = [earlystopping_callback])

    

    loss = history.history["loss"]

    val_loss = history.history["val_loss"]

    mae = history.history["mean_absolute_error"]

    val_mae = history.history["val_mean_absolute_error"]

    

    all_loss.append(loss)

    all_val_loss.append(val_loss)

    all_mae.append(mae)

    all_val_mae.append(val_mae)



average_all_loss = np.mean([i[-1] for i in all_loss])

average_all_val_loss = np.mean([i[-1] for i in all_val_loss])

average_all_mae = np.mean([i[-1] for i in all_mae])

average_all_val_mae = np.mean([i[-1] for i in all_val_mae])



print("Loss: {}, Val_Loss: {}, MAE: {}, Val_MAE: {}".format(average_all_loss, average_all_val_loss, average_all_mae, average_all_val_mae))

'''
model = build_model()

earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience=20, verbose=1, mode='auto')

history = model.fit(x = X_train, y = Y_train, epochs = epochs, batch_size = 100, validation_data = (X_test, Y_test), callbacks = [earlystopping_callback])
model = build_model()

earlystopping_callback = EarlyStopping(monitor = 'val_loss', patience=50, verbose=1, mode='auto')

history = model.fit(x = na_train_x, y = na_train_y, epochs = epochs, batch_size = 100, validation_split = 0.1, callbacks = [earlystopping_callback])
predict = model.predict(na_test_x)
na_output = np.concatenate([na_test_id, predict], axis = 1)

output = pd.DataFrame(na_output.astype(np.uint32), columns = ["Id", "SalePrice"])

output.to_csv("submission.csv", index = False)

output.head(300)