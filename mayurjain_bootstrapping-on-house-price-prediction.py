import numpy as np

import pandas as pd



from sklearn.preprocessing import normalize,MinMaxScaler,StandardScaler

from sklearn.decomposition import PCA, randomized_svd

from sklearn import random_projection



from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from lightgbm import LGBMRegressor

from sklearn.svm import SVR

from sklearn.ensemble import GradientBoostingRegressor,VotingRegressor,BaggingRegressor, RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV

from catboost import CatBoostRegressor



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV



import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



%matplotlib inline
training_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

testing_set = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

sample_submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
print("Training Set: ",training_set.shape)

print("Testing Set ",testing_set.shape)
with open("/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt","r") as data_desc:

    for line in data_desc.readlines()[:10]:

        print(line)
y = training_set["SalePrice"]
def check_null(df,col):

    return print(col," ",df[col].isnull().sum())
def categorical_saleprice(df,col):

    return df.groupby(col)["SalePrice"].mean()
columns_impute_train = training_set.columns[(training_set.isnull().sum() > 0) & (training_set.isnull().sum() < 100)]

columns_impute_test = testing_set.columns[(testing_set.isnull().sum() > 0) & (testing_set.isnull().sum() < 100)]
#Numeric Columns

num_cols = training_set._get_numeric_data().columns
#training_set[columns_impute_train[0]].fillna(training_set[columns_impute_train[0]].mode()[0],inplace=True)
print(columns_impute_train)

print(columns_impute_test)
check_null(training_set,columns_impute_train[0])

categorical_saleprice(training_set,columns_impute_train[0])
MasVnrType = list(training_set.groupby(columns_impute_train[0])["SalePrice"].mean().index)

sale_price = list(training_set.groupby(columns_impute_train[0])["SalePrice"].mean().values)
#plt.bar(MasVnrType,sale_price)
training_set[training_set[columns_impute_train[0]].isnull()]
training_set.loc[training_set['Id']==235,'MasVnrType'] = "BrkFace"

training_set.loc[training_set['Id']==530,'MasVnrType'] = "BrkFace"

training_set.loc[training_set['Id']==651,'MasVnrType'] = "BrkFace"

training_set.loc[training_set['Id']==937,'MasVnrType'] = "None"

training_set.loc[training_set['Id']==974,'MasVnrType'] = "None"

training_set.loc[training_set['Id']==978,'MasVnrType'] = "BrkFace"

training_set.loc[training_set['Id']==1244,'MasVnrType'] = "Stone"

training_set.loc[training_set['Id']==1279,'MasVnrType'] = "Stone"
print(training_set.groupby(columns_impute_train[1])["SalePrice"].mean())

print(training_set.groupby(columns_impute_train[0])["MasVnrArea"].value_counts())

print(training_set[(training_set["MasVnrType"]=="Stone") & (training_set["SalePrice"] >= 230000)][["MasVnrArea","SalePrice"]].mean())
training_set[(training_set["MasVnrType"]=="BrkFace") & (training_set["SalePrice"] >= 200000)][["MasVnrArea","SalePrice"]].mean()
training_set[training_set[columns_impute_train[1]].isnull()][["MasVnrType","SalePrice"]]
training_set.loc[training_set['Id']==937,'MasVnrArea'] = 0.0

training_set.loc[training_set['Id']==974,'MasVnrArea'] = 0.0

training_set.loc[training_set['Id']==978,'MasVnrArea'] = 332.00

training_set.loc[training_set['Id']==1244,'MasVnrArea'] = 280.3

training_set.loc[training_set['Id']==1279,'MasVnrArea'] = 280.3

training_set.loc[training_set['Id']==235,'MasVnrArea'] = 332.00

training_set.loc[training_set['Id']==530,'MasVnrArea'] = 332.00

training_set.loc[training_set['Id']==651,'MasVnrArea'] = 332.00
training_set[columns_impute_train[2]].value_counts()
training_set[training_set[columns_impute_train[2]].isnull()].head()
print(training_set.groupby(columns_impute_train[2])["SalePrice"].mean())
print(training_set[(training_set["BsmtQual"]=="Ex") ][["SalePrice"]].mean())

print(training_set[(training_set["BsmtQual"]=="TA") ][["SalePrice"]].mean())

print(training_set[(training_set["BsmtQual"]=="Fa") ][["SalePrice"]].mean())
def impute_BsmtQual_train(df,column):

    df.loc[(df[column].isnull()) & (df["SalePrice"] < 120000.00),column] = "Fa"

    df.loc[(df[column].isnull()) & (df["SalePrice"] >= 120000.00) & (df["SalePrice"] < 180000.00),column] = "TA"

    df.loc[(df[column].isnull()) & (df["SalePrice"] >= 180000.00),column] = "TA"

    return "Basement Quality's missing value is imputed"
impute_BsmtQual_train(training_set,"BsmtQual")
training_set[columns_impute_train[3]].value_counts()
training_set[training_set[columns_impute_train[3]].isnull()].head()
#print(training_set[(training_set["BsmtCond"]=="Ex") ][["SalePrice"]].mean())

print(training_set[(training_set["BsmtCond"]=="TA") ][["SalePrice"]].mean())

print(training_set[(training_set["BsmtCond"]=="Fa") ][["SalePrice"]].mean())

print(training_set[(training_set["BsmtCond"]=="Po") ][["SalePrice"]].mean())

print(training_set[(training_set["BsmtCond"]=="Gd") ][["SalePrice"]].mean())
print(training_set.groupby(columns_impute_train[3])["SalePrice"].mean())
training_set.loc[(training_set[columns_impute_train[3]].isnull()) & (training_set["SalePrice"] < 75000.00),"BsmtCond"] = "Po"

training_set.loc[(training_set[columns_impute_train[3]].isnull()) & (training_set["SalePrice"] >= 75000.00) & (training_set["SalePrice"] < 140000.00),"BsmtCond"] = "Fa"

training_set.loc[(training_set[columns_impute_train[3]].isnull()) & (training_set["SalePrice"] >= 140000.00) & (training_set["SalePrice"] < 190000.00),"BsmtCond"] = "TA"

training_set.loc[(training_set[columns_impute_train[3]].isnull()) & (training_set["SalePrice"] >= 190000.00),"BsmtCond"] = "Gd"
training_set[columns_impute_train[4]].value_counts()
training_set[training_set[columns_impute_train[4]].isnull()].shape
print(training_set.groupby(columns_impute_train[4])["SalePrice"].mean())
training_set.loc[(training_set[columns_impute_train[4]].isnull()) & (training_set["SalePrice"] < 150000.00),"BsmtExposure"] = "No"

training_set.loc[(training_set[columns_impute_train[4]].isnull()) & (training_set["SalePrice"] >= 150000.00) & (training_set["SalePrice"] < 200000.00),"BsmtExposure"] = "Mn"

training_set.loc[(training_set[columns_impute_train[4]].isnull()) & (training_set["SalePrice"] >= 200000.00) & (training_set["SalePrice"] < 230000.00),"BsmtExposure"] = "Av"

training_set.loc[(training_set[columns_impute_train[4]].isnull()) & (training_set["SalePrice"] >= 210000.00),"BsmtExposure"] = "Gd"
training_set[columns_impute_train[5]].value_counts()
training_set[training_set[columns_impute_train[5]].isnull()].shape
print(training_set.groupby(columns_impute_train[5])["SalePrice"].mean())
training_set[training_set[columns_impute_train[5]].isnull()][["BsmtCond","BsmtExposure","BsmtFinType1","BsmtQual"]].head()
training_set.loc[(training_set[columns_impute_train[5]].isnull()) & (training_set["BsmtQual"] == "Fa"),"BsmtFinType1"] = "Unf"

training_set.loc[(training_set[columns_impute_train[5]].isnull()) & (training_set["BsmtQual"] == "TA"),"BsmtFinType1"] = "LwQ"

training_set.loc[(training_set[columns_impute_train[5]].isnull()) & (training_set["BsmtQual"] == "Gd"),"BsmtFinType1"] = "GLQ"
training_set[columns_impute_train[6]].value_counts()
training_set[training_set[columns_impute_train[6]].isnull()].shape
print(training_set.groupby(columns_impute_train[6])["SalePrice"].mean())
training_set[training_set["BsmtQual"]=="Ex"]["BsmtFinType2"].value_counts()
training_set.loc[training_set[columns_impute_train[6]].isnull(),"BsmtFinType2"] = "Unf"
training_set[columns_impute_train[7]].value_counts()
training_set[columns_impute_train[7]].fillna(value="SBrkr",inplace=True)
training_set[columns_impute_train[8]].value_counts()
training_set[training_set[columns_impute_train[8]].isnull()]
print(training_set.groupby(columns_impute_train[8])["SalePrice"].mean())
training_set.loc[(training_set[columns_impute_train[8]].isnull()) & (training_set["SalePrice"] < 110000.00),"GarageType"] = "CarPort"

training_set.loc[(training_set[columns_impute_train[8]].isnull()) & (training_set["SalePrice"] >= 110000.00) & (training_set["SalePrice"] < 140000.00),"GarageType"] = "Detchd"

training_set.loc[(training_set[columns_impute_train[8]].isnull()) & (training_set["SalePrice"] >= 140000.00) & (training_set["SalePrice"] < 210000.00),"GarageType"] = "Attchd"

training_set.loc[(training_set[columns_impute_train[8]].isnull()) & (training_set["SalePrice"] >= 210000.00),"GarageType"] = "BuiltIn"
#training_set[columns_impute_train[9]].value_counts()
#training_set[training_set[columns_impute_train[9]].isnull()].shape
#print(training_set.groupby(columns_impute_train[9])["SalePrice"].mean())
#pd.crosstab(training_set["GarageYrBlt"],training_set["GarageType"]).ipynb_checkpoints/
#training_set[training_set["GarageYrBlt"].isnull()]["GarageType"].value_counts()
#training_set[training_set["GarageType"]=="Detchd"]["GarageYrBlt"].value_counts()
columns_impute_train.drop(columns_impute_train[9])
training_set[columns_impute_train[10]].value_counts()
training_set[training_set[columns_impute_train[10]].isnull()]
print(training_set.groupby(columns_impute_train[10])["SalePrice"].mean())
pd.crosstab(training_set["SalePrice"],training_set["GarageFinish"])
training_set.loc[(training_set[columns_impute_train[10]].isnull()) & (training_set["SalePrice"] < 160000.00),"GarageFinish"] = "Unf"

training_set.loc[(training_set[columns_impute_train[10]].isnull()) & (training_set["SalePrice"] >= 160000.00) & (training_set["SalePrice"] < 220000.00),"GarageFinish"] = "RFn"

training_set.loc[(training_set[columns_impute_train[10]].isnull()) & (training_set["SalePrice"] >= 220000.00),"GarageFinish"] = "Fin"
training_set[columns_impute_train[11]].value_counts()
training_set[training_set[columns_impute_train[11]].isnull()].head()
print(training_set.groupby(columns_impute_train[11])["SalePrice"].mean())
training_set.loc[(training_set[columns_impute_train[11]].isnull()) & (training_set["SalePrice"] < 100000.00),"GarageQual"] = "Po"

training_set.loc[(training_set[columns_impute_train[11]].isnull()) & (training_set["SalePrice"] >= 100000.00) & (training_set["SalePrice"] < 140000.00),"GarageQual"] = "Fa"

training_set.loc[(training_set[columns_impute_train[11]].isnull()) & (training_set["SalePrice"] >= 140000.00) & (training_set["SalePrice"] < 220000.00),"GarageQual"] = "Ta"

training_set.loc[(training_set[columns_impute_train[11]].isnull()) & (training_set["SalePrice"] >= 220000.00),"GarageQual"] = "Ex"
training_set[columns_impute_train[12]].value_counts()
training_set[training_set[columns_impute_train[12]].isnull()].shape
print(training_set.groupby(columns_impute_train[12])["SalePrice"].mean())
training_set[columns_impute_train[12]].fillna(value="TA",inplace=True)
columns_too_many_missing = list(training_set.columns[training_set.isnull().any()])
list(training_set.columns[training_set.isnull().any()])
training_set.drop(columns=columns_too_many_missing,inplace=True)
testing_set.drop(columns=columns_too_many_missing,inplace=True)
training_set.select_dtypes(include='object').columns

col_to_encode = ['MSZoning', 'Street', 'LotShape', 'LandContour',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

       'PavedDrive', 'SaleType', 'SaleCondition']
encoded_categorical_df = pd.get_dummies(training_set, columns = col_to_encode, drop_first = True)
col_drop = list(training_set.select_dtypes(include='object').columns)

col_drop.append("Id")

training_set.drop(columns=col_drop,inplace=True)
encoded_categorical_df.drop(columns=["Id","Utilities"],inplace = True)
min_max_scaler = MinMaxScaler()

standard_scaler = StandardScaler()

col_to_normalize = ['MSSubClass', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea'

        ,'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']

encoded_categorical_df[col_to_normalize] = min_max_scaler.fit_transform(encoded_categorical_df[col_to_normalize])

#encoded_categorical_df[col_to_normalize] = standard_scaler.fit_transform(encoded_categorical_df[col_to_normalize])

#encoded_categorical_df[col_to_normalize] = normalize(encoded_categorical_df[col_to_normalize])
encoded_categorical_df[['MSSubClass', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF','LowQualFinSF', 'GrLivArea'

        ,'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']]
#encoded_categorical_df.drop(columns=col_to_normalize,inplace=True)
encoded_categorical_df.drop(columns=["SalePrice"],inplace=True)
columns_impute_test
testing_set["MSZoning"].value_counts()
testing_set["MSSubClass"].value_counts()
testing_set[testing_set[columns_impute_test[0]].isnull()]
testing_set[testing_set[columns_impute_test[0]].isnull()][["MSSubClass","MSZoning"]]
pd.crosstab(testing_set["MSSubClass"],testing_set["MSZoning"])
testing_set.loc[(testing_set[columns_impute_test[0]].isnull()) & (testing_set["MSSubClass"] == 20),"MSZoning"] = "RL"

testing_set.loc[(testing_set[columns_impute_test[0]].isnull()) & (testing_set["MSSubClass"] == 30),"MSZoning"] = "RM"

testing_set.loc[(testing_set[columns_impute_test[0]].isnull()) & (testing_set["MSSubClass"] == 70),"MSZoning"] = "RM"

testing_set[columns_impute_test[1]].value_counts()
testing_set[testing_set[columns_impute_test[1]].isnull()]
testing_set[columns_impute_test[1]].fillna(value="AllPub",inplace=True)
testing_set[columns_impute_test[2]].value_counts()
testing_set[testing_set[columns_impute_test[2]].isnull()]
testing_set[columns_impute_test[2]].fillna(value="VinylSd",inplace=True)
testing_set[columns_impute_test[3]].value_counts()
testing_set[testing_set[columns_impute_test[3]].isnull()]
testing_set[columns_impute_test[3]].fillna(value="VinylSd",inplace=True)
testing_set[columns_impute_test[4]].value_counts()
testing_set[testing_set[columns_impute_test[4]].isnull()][["MasVnrType","ExterCond","ExterQual"]]
testing_set[testing_set["ExterQual"]=="Gd"]["MasVnrType"].value_counts()
pd.crosstab(testing_set["MasVnrType"],testing_set["ExterCond"])
testing_set["MasVnrType"].fillna(value="None",inplace=True)
testing_set[columns_impute_test[5]].value_counts()
testing_set[testing_set[columns_impute_test[5]].isnull()]
testing_set[columns_impute_test[5]].fillna(value=0.0,inplace=True)
testing_set[columns_impute_test[6]].value_counts()
testing_set[testing_set[columns_impute_test[6]].isnull()].head()
pd.crosstab(testing_set["BsmtQual"],testing_set["GarageQual"])
testing_set[testing_set[columns_impute_test[6]].isnull()][["BsmtQual","GarageQual"]].head()
testing_set.loc[(testing_set[columns_impute_test[6]].isnull()) & (testing_set["GarageQual"] == "Ta"),"BsmtQual"] = "Gd"

testing_set[columns_impute_test[6]].fillna(value="TA",inplace=True)
testing_set[columns_impute_test[7]].value_counts()
testing_set[columns_impute_test[7]].fillna(value="TA",inplace=True)
testing_set[columns_impute_test[8]].value_counts()
testing_set[testing_set[columns_impute_test[8]].isnull()][["BsmtExposure","GarageQual"]].head()
testing_set[testing_set["GarageQual"]=="TA"]["BsmtExposure"].value_counts()
testing_set[columns_impute_test[8]].fillna(value="No",inplace=True)
testing_set[columns_impute_test[9]].value_counts()
testing_set[testing_set[columns_impute_test[9]].isnull()].head()
testing_set[testing_set[columns_impute_test[9]].isnull()][["BsmtFinType1","GarageCond","GarageQual","BldgType"]].head()
testing_set[testing_set["BldgType"]=="1Fam"]["BsmtFinType1"].value_counts()
testing_set.loc[(testing_set[columns_impute_test[9]].isnull()) & (testing_set["BldgType"] == "1Fam"),"BsmtFinType1"] = "Unf"

testing_set.loc[(testing_set[columns_impute_test[9]].isnull()) & (testing_set["BldgType"] == "Duplex"),"BsmtFinType1"] = "GLQ"

testing_set.columns
testing_set[columns_impute_test[10]].value_counts()
testing_set[testing_set[columns_impute_test[10]].isnull()]
testing_set[testing_set["SaleCondition"]=="Abnorml"][columns_impute_test[10]].value_counts().head()
testing_set[columns_impute_test[10]].fillna(value=0.0,inplace=True)
testing_set[columns_impute_test[11]].value_counts()
testing_set[testing_set[columns_impute_test[11]].isnull()].head()
testing_set[testing_set["SaleCondition"]=="AdjLand"][columns_impute_test[11]].value_counts()
testing_set[columns_impute_test[11]].fillna(value="Unf",inplace=True)

testing_set[columns_impute_test[12]].fillna(value=0.0,inplace=True)
testing_set[columns_impute_test[13]].value_counts()
testing_set[testing_set[columns_impute_test[13]].isnull()][["BsmtFinType1","GarageCond","GarageQual","BldgType"]]
testing_set[columns_impute_test[13]].fillna(value=0.0,inplace=True)
testing_set[columns_impute_test[14]].value_counts()
testing_set[testing_set[columns_impute_test[14]].isnull()][["BsmtFinType1","GarageCond","GarageQual","BldgType"]]
testing_set[columns_impute_test[14]].fillna(value=0.0,inplace=True)
testing_set[columns_impute_test[15]].value_counts()
testing_set[testing_set[columns_impute_test[15]].isnull()][["BsmtFinType1","GarageCond","GarageQual","BldgType"]]
testing_set[columns_impute_test[15]].fillna(value=0.0,inplace=True)
testing_set[columns_impute_test[16]].value_counts()
testing_set[columns_impute_test[16]].fillna(value=0.0,inplace=True)
testing_set[columns_impute_test[17]].value_counts()
testing_set[testing_set[columns_impute_test[17]].isnull()][["BsmtFinType1","GarageCond","GarageQual","BldgType"]]
testing_set[columns_impute_test[17]].fillna(value="Fa",inplace=True)
testing_set[columns_impute_test[18]].value_counts()
testing_set[testing_set[columns_impute_test[18]].isnull()][["BsmtFinType1","GarageCond","GarageQual","BldgType"]]
testing_set[columns_impute_test[18]].fillna(value="Fa",inplace=True)
testing_set[columns_impute_test[19]].value_counts()
testing_set[testing_set[columns_impute_test[19]].isnull()][["BldgType","GarageType"]]
testing_set[testing_set["BldgType"]=="TwnhsE"][columns_impute_test[19]].value_counts()
testing_set.loc[(testing_set[columns_impute_test[19]].isnull()) & (testing_set["BldgType"] == "1Fam"),"GarageType"] = "Attchd"

testing_set.loc[(testing_set[columns_impute_test[19]].isnull()) & (testing_set["BldgType"] == "Duplex"),"GarageType"] = "Detchd"

testing_set.loc[(testing_set[columns_impute_test[19]].isnull()) & (testing_set["BldgType"] == "2fmCon"),"GarageType"] = "Detchd"

testing_set.loc[(testing_set[columns_impute_test[19]].isnull()) & (testing_set["BldgType"] == "Twnhs"),"GarageType"] = "Detchd"

testing_set.loc[(testing_set[columns_impute_test[19]].isnull()) & (testing_set["BldgType"] == "TwnhsE"),"GarageType"] = "Attchd"
testing_set[columns_impute_test[21]].value_counts()
testing_set[testing_set[columns_impute_test[21]].isnull()].head()
testing_set[testing_set[columns_impute_test[21]].isnull()]["BldgType"].value_counts()
testing_set.loc[(testing_set[columns_impute_test[21]].isnull()) & (testing_set["BldgType"] == "TwnhsE"),"GarageFinish"] = "Fin"

testing_set[columns_impute_test[21]].fillna(value="Unf",inplace=True)
testing_set[columns_impute_test[22]].value_counts()
testing_set[testing_set[columns_impute_test[22]].isnull()]
testing_set[testing_set["SaleCondition"]=="Alloca"][columns_impute_test[22]].value_counts()
testing_set[columns_impute_test[22]].fillna(value=2.0,inplace=True)
testing_set[columns_impute_test[23]].value_counts()
testing_set[testing_set[columns_impute_test[23]].isnull()]
testing_set[testing_set["SaleCondition"]=="Alloca"][columns_impute_test[23]].mean()
testing_set[columns_impute_test[23]].fillna(value=499.09,inplace=True)
testing_set[columns_impute_test[24]].value_counts()
testing_set[testing_set[columns_impute_test[24]].isnull()].head()
testing_set[columns_impute_test[24]].fillna(value="TA",inplace=True)
testing_set[columns_impute_test[25]].value_counts()
testing_set[columns_impute_test[25]].fillna(value="TA",inplace=True)
testing_set[columns_impute_test[26]].value_counts()
testing_set[testing_set[columns_impute_test[26]].isnull()]
testing_set[testing_set["SaleCondition"]=="Normal"][columns_impute_test[26]].value_counts()
testing_set[columns_impute_test[26]].fillna(value="WD",inplace=True)
testing_set.columns[(testing_set.isnull().sum() > 0) & (testing_set.isnull().sum() < 100)]
testing_set.drop(columns=["Id"],inplace=True)

testing_set.drop(columns=["Utilities"],inplace=True)
col_for_dummies = ["MSZoning","Street","LotShape","LandContour","LotConfig","LandSlope","Neighborhood","Condition1","Condition2","BldgType",

                  "HouseStyle","RoofStyle",'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical',

                   'KitchenQual', 'Functional', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition']
testing_encoding_categorical = pd.get_dummies(testing_set, columns = col_for_dummies, drop_first = True)
#min_max_scaler = MinMaxScaler()

num_cols_test = testing_set._get_numeric_data().columns

num_cols_test
col_to_normalize = ["MSSubClass","LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF",

                   "GrLivArea","GarageArea","WoodDeckSF","OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"]

testing_encoding_categorical[col_to_normalize] = min_max_scaler.transform(testing_encoding_categorical[col_to_normalize])

#testing_encoding_categorical[col_to_normalize] = standard_scaler.transform(testing_encoding_categorical[col_to_normalize])

#testing_encoding_categorical[col_to_normalize] = normalize(testing_encoding_categorical[col_to_normalize])
#testing_encoding_categorical.drop(columns=col_to_normalize,inplace=True)
features_to_delete_test = []

for col in testing_encoding_categorical.columns:

    if col not in encoded_categorical_df.columns:

        features_to_delete_test.append(col)
features_to_delete_train = []

for col in encoded_categorical_df.columns:

    if col not in testing_encoding_categorical.columns:

        features_to_delete_train.append(col)
print(encoded_categorical_df.shape)

print(testing_encoding_categorical.shape)
#features_to_delete_train.remove("SalePrice")
encoded_categorical_df.drop(columns=features_to_delete_train,inplace=True)
testing_encoding_categorical.drop(columns=features_to_delete_test,inplace=True)
features_to_delete_train
#grp = PCA(n_components=)

#X = grp.fit_transform(encoded_categorical_df)

#print(type(X))

#print(type(y))
#grp = random_projection.johnson_lindenstrauss_min_dim(encoded_categorical_df,eps=0.3)
X_train, X_valid, y_train, y_valid  = train_test_split(encoded_categorical_df,y,test_size=0.25)
cbr = CatBoostRegressor(logging_level='Silent')

cbr.fit(X_train,y_train)

y_pred = cbr.predict(X_valid)

print("Mean Squared Error",mean_squared_error(y_valid, y_pred))

print("Mean Absolute Error",mean_absolute_error(y_valid, y_pred))

print("R2 Score",r2_score(y_valid, y_pred))
# param_grid = {'learning_rate':[0.1],"n_estimators":[150],'min_samples_leaf':[5],'min_samples_split':[20],"max_depth":[10],'loss': ['lad'],"max_features":["auto"]}

# gbccv = GridSearchCV(GradientBoostingRegressor(),param_grid)

# gbccv.fit(X_train,y_train)

# y_pred = gbccv.predict(X_valid)

# print("Mean Squared Error",mean_squared_error(y_valid, y_pred))

# print("Mean Absolute Error",mean_absolute_error(y_valid, y_pred))

# print("R2 Score",r2_score(y_valid, y_pred))
# param_grid = {'learning_rate':[0.1],"n_estimators":[100],"min_split_gain":[0.1],"min_child_samples":[15]}

# lgbm = GridSearchCV(LGBMRegressor(),param_grid)

# lgbm.fit(X_train,y_train)

# y_pred = lgbm.predict(X_valid)

# print(lgbm.best_params_)

# print(lgbm.best_score_)

# print("Mean Squared Error",mean_squared_error(y_valid, y_pred))

# print("Mean Absolute Error",mean_absolute_error(y_valid, y_pred))

# print("R2 Score",r2_score(y_valid, y_pred))
# reg1 = GradientBoostingRegressor()

# reg2 = LGBMRegressor()

# reg3 = CatBoostRegressor()

# voting_regressor = VotingRegressor(estimators=[('gbr', reg1), ('lgbm', reg2),('catboost',reg3)])

# voting_regressor.fit(X_train,y_train)

# y_pred = voting_regressor.predict(X_valid)

# print("Mean Squared Error",mean_squared_error(y_valid, y_pred))

# print("Mean Absolute Error",mean_absolute_error(y_valid, y_pred))

# print("R2 Score",r2_score(y_valid, y_pred))
# las_reg = RidgeCV(cv=10).fit(X_train,y_train)

# y_pred = las_reg.predict(X_valid)

# print("Mean Squared Error",mean_squared_error(y_valid, y_pred))

# print("Mean Absolute Error",mean_absolute_error(y_valid, y_pred))

# print("R2 Score",r2_score(y_valid, y_pred))
#testing_encoding_categorical_pca = pca.transform(testing_encoding_categorical)
Id = sample_submission["Id"]

predicted_test = []

#X_test = random_projection.johnson_lindenstrauss_min_dim(testing_encoding_categorical,eps=0.3)

for x in cbr.predict(testing_encoding_categorical):

    predicted_test.append(x)

predicted_test_value = pd.DataFrame({ 'Id': Id,

                        'SalePrice': predicted_test })

predicted_test_value.to_csv("PredictedTestScore.csv", index=False)