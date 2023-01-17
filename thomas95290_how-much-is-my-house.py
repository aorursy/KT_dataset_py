import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import cross_val_score

import seaborn as sns

from sklearn.preprocessing import StandardScaler, normalize

import matplotlib

import matplotlib.pyplot as plt

import xgboost as xgb

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
#I use log so i can compare result with the leaderboard 

y = np.log(train["SalePrice"])

#I delete the saleprice column so I can concatenate the train and test set

train.drop("SalePrice", axis=1, inplace=True)

data = pd.concat([train,test])

data = data.set_index("Id")
fig, axs = plt.subplots(11,4, figsize=(13, 40))

sns.countplot(data.MSSubClass, ax=axs[0,0])

sns.countplot(data.MSZoning, ax=axs[0,1])

sns.countplot(data.Street, ax=axs[0,2])

sns.countplot(data.Alley, ax=axs[0,3])

sns.countplot(data.LotShape, ax=axs[1,0])

sns.countplot(data.LandContour, ax=axs[1,1])

sns.countplot(data.Utilities, ax=axs[1,2])

sns.countplot(data.LotConfig, ax=axs[1,3])

sns.countplot(data.LandSlope, ax=axs[2,0])

sns.countplot(data.Condition1, ax=axs[2,1])

sns.countplot(data.Condition2, ax=axs[2,2])

sns.countplot(data.BldgType, ax=axs[2,3])

sns.countplot(data.HouseStyle, ax=axs[3,0])

sns.countplot(data.OverallQual, ax=axs[3,1])

sns.countplot(data.OverallCond, ax=axs[3,2])

sns.countplot(data.RoofStyle, ax=axs[3,3])

sns.countplot(data.RoofMatl, ax=axs[4,0])

sns.countplot(data.MasVnrType, ax=axs[4,1])

sns.countplot(data.ExterQual, ax=axs[4,2])

sns.countplot(data.ExterCond, ax=axs[4,3])

sns.countplot(data.Foundation, ax=axs[5,0])

sns.countplot(data.BsmtQual, ax=axs[5,1])

sns.countplot(data.BsmtCond, ax=axs[5,2])

sns.countplot(data.BsmtExposure, ax=axs[5,3])

sns.countplot(data.BsmtFinType1, ax=axs[6,0])

sns.countplot(data.BsmtFinType2, ax=axs[6,1])

sns.countplot(data.Heating, ax=axs[6,2])

sns.countplot(data.HeatingQC, ax=axs[6,3])

sns.countplot(data.CentralAir, ax=axs[7,0])

sns.countplot(data.Electrical, ax=axs[7,1])

sns.countplot(data.KitchenQual, ax=axs[7,2])

sns.countplot(data.Functional, ax=axs[7,3])

sns.countplot(data.FireplaceQu, ax=axs[8,0])

sns.countplot(data.GarageType, ax=axs[8,1])

sns.countplot(data.GarageFinish, ax=axs[8,2])

sns.countplot(data.GarageQual, ax=axs[8,3])

sns.countplot(data.GarageCond, ax=axs[9,0])

sns.countplot(data.PavedDrive, ax=axs[9,1])

sns.countplot(data.PoolQC, ax=axs[9,2])

sns.countplot(data.Fence, ax=axs[9,3])

sns.countplot(data.MiscFeature, ax=axs[10,0])

sns.countplot(data.SaleType, ax=axs[10,1])

sns.countplot(data.SaleCondition, ax=axs[10,2])
fig, axs = plt.subplots(3,1, figsize=(13, 8))



sns.countplot(data.Neighborhood, ax=axs[0])

sns.countplot(data.Exterior1st, ax=axs[1])

sns.countplot(data.Exterior2nd, ax=axs[2])
data.MSZoning = (data.MSZoning == "RL").astype(int)

data.LotShape = (data.LotShape == "Reg").astype(int) 

data.LandContour = (data.LandContour == "Lvl").astype(int) 

data.LotConfig = (data.LotConfig == "Inside").astype(int) 

data.LandSlope = (data.LandSlope == "Gtl").astype(int) 

data.Condition1 = (data.Condition1 == "Norm").astype(int) 

data.Condition2 = (data.Condition2 == "Norm").astype(int) 

data.BldgType = (data.BldgType == "1Fam").astype(int) 

data.RoofStyle = (data.RoofStyle == "Gable").astype(int) 

data.RoofMatl = (data.RoofMatl == "CompShg").astype(int) 

data.ExterCond = (data.ExterCond == "TA").astype(int)

data.BsmtCond = (data.BsmtCond == "TA").astype(int)

data.BsmtFinType2 = (data.BsmtFinType2 == "Unf").astype(int)

data.Heating = (data.Heating == "GasA").astype(int)

data.Electrical = (data.Electrical == "SBrkr").astype(int)

data.Functional = (data.Functional == "Typ").astype(int)

data.GarageQual = (data.GarageQual == "TA").astype(int)

data.GarageCond = (data.GarageCond == "TA").astype(int)

data.PavedDrive = (data.PavedDrive == "Y").astype(int)

data.SaleType = (data.SaleType == "WD").astype(int)

data.SaleCondition = (data.SaleCondition == "Normal").astype(int)
data.head()
data.drop("PoolQC", axis=1, inplace=True)

data.drop("Fence", axis=1, inplace=True)

data.drop("MiscFeature", axis=1, inplace=True)

data.drop("Alley", axis = 1, inplace=True)
#all the quantitave variables are collect with the describe function

quant_variable = data.describe().columns.values

column = data.columns.values
for i in column:

    if i not in quant_variable:

        #we are with qualitative variable

        data[i].fillna("no_present", inplace=True)

        dummy_variable = pd.get_dummies(data[i], prefix=i)

        data = data.join(dummy_variable)

        data.drop(i, axis=1, inplace=True)
data.head()
data = data.fillna(data.mean())

#Scaling the data

data = (data - data.mean()) / (data.std())
#We resplit the data in train and test now

X_train = data[:train.shape[0]]

X_test = data[train.shape[0]:]
#Function to calcul the error of our models

def rmse_cv(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="mean_squared_error", cv = 5))

    return(rmse)
gb = GradientBoostingRegressor(random_state=10, n_estimators=100, max_depth=2, learning_rate=0.15).fit(X_train, y)

print (rmse_cv(gb).mean())
gbm = xgb.XGBRegressor(max_depth=3, n_estimators=1000, learning_rate=0.1,

	objective='reg:linear').fit(X_train, y)

print (rmse_cv(gbm).mean())