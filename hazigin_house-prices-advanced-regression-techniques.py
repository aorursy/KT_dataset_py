#Load Library
import pandas as pd
import numpy as np
import seaborn as sns
#Load Competition's Datas
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
#Concat Train Data And Test Data
df_temp = pd.concat([df_train,df_test])
#Drop Unused Columns
df_temp.drop(["MiscFeature","MiscVal","PoolQC","PoolArea","Fireplaces","FireplaceQu","Fence","Alley"],axis=1,inplace=True)
df_temp["MasVnrType"].unique()
df_temp["MasVnrType"][df_temp["MasVnrType"].isnull()==True]="None"
df_temp["MasVnrArea"][df_temp["MasVnrArea"].isnull()==True] = df_train["MasVnrArea"].mean()
df_temp["BsmtQual"].value_counts()
df_temp["BsmtQual"][df_temp["BsmtQual"].isnull()==True] = "TA"
df_temp["BsmtCond"].value_counts()
df_temp["BsmtCond"][df_temp["BsmtCond"].isnull()==True] = "TA"
df_temp["BsmtExposure"].value_counts()
df_temp["BsmtExposure"][df_temp["BsmtExposure"].isnull()==True] = "No"
df_temp["BsmtFinType1"].value_counts()
df_temp["BsmtFinType1"][df_temp["BsmtFinType1"].isnull()==True] = "Unf"
df_temp["BsmtFinSF1"].value_counts()
df_temp["BsmtFinSF1"][df_temp["BsmtFinSF1"].isnull()==True] = 0
df_temp["BsmtFinSF2"].value_counts()
df_temp["BsmtFinSF2"][df_temp["BsmtFinSF2"].isnull()==True] = 0
df_temp["BsmtFinType2"].value_counts()
df_temp["BsmtFinType2"][df_temp["BsmtFinType2"].isnull()==True] = "Unf"
df_temp["BsmtFullBath"].value_counts()
df_temp["BsmtFullBath"][df_temp["BsmtFullBath"].isnull()==True] = 0
df_temp["BsmtHalfBath"].value_counts()
df_temp["BsmtHalfBath"][df_temp["BsmtHalfBath"].isnull()==True] = 0
df_temp["BsmtUnfSF"].value_counts()
df_temp["BsmtUnfSF"][df_temp["BsmtUnfSF"].isnull()==True] = 0
df_temp["Exterior1st"].value_counts()
df_temp["Exterior1st"][df_temp["Exterior1st"].isnull()==True] = "VinylSd"
df_temp["Exterior2nd"].value_counts()
df_temp["Exterior2nd"][df_temp["Exterior2nd"].isnull()==True] = "VinylSd"
df_temp["Functional"].value_counts()
df_temp["Functional"][df_temp["Functional"].isnull()==True] = "Typ"
df_temp["GarageArea"].value_counts()
df_temp["GarageArea"][df_temp["GarageArea"].isnull()==True] = 576
df_temp["GarageCars"].value_counts()
df_temp["GarageCars"][df_temp["GarageCars"].isnull()==True] = 2
df_temp["KitchenQual"].value_counts()
df_temp["KitchenQual"][df_temp["KitchenQual"].isnull()==True] = "TA"
df_temp["LotFrontage"].value_counts()
df_temp["LotFrontage"][df_temp["LotFrontage"].isnull()==True] = 60
df_temp["MSZoning"].value_counts()
df_temp["MSZoning"][df_temp["MSZoning"].isnull()==True] = "RL"
df_temp["SaleType"].value_counts()
df_temp["SaleType"][df_temp["SaleType"].isnull()==True] = "WD"
df_temp["TotalBsmtSF"].value_counts()
df_temp["TotalBsmtSF"][df_temp["TotalBsmtSF"].isnull()==True] = 0
df_temp["Utilities"].value_counts()
df_temp["Utilities"][df_temp["Utilities"].isnull()==True] = "AllPub"
df_temp["Electrical"].value_counts()
df_temp["Electrical"][df_temp["Electrical"].isnull()==True] = "SBrkr"
df_temp["GarageType"].value_counts()
df_temp["GarageType"][df_temp["GarageType"].isnull()==True] = "Attchd"
df_temp["GarageYrBlt"].value_counts()
df_temp["GarageYrBlt"][df_temp["GarageYrBlt"].isnull()==True] = df_temp["GarageYrBlt"][df_temp["GarageYrBlt"] > 2000].mean()
df_temp["GarageFinish"].value_counts()
df_temp["GarageFinish"][df_temp["GarageFinish"].isnull()==True] = "Unf"
df_temp["GarageQual"].value_counts()
df_temp["GarageQual"][df_temp["GarageQual"].isnull()==True] = "TA"
df_temp["GarageCond"].value_counts()
df_temp["GarageCond"][df_temp["GarageCond"].isnull()==True] = "TA"
#Check Are There NAN yet
df_temp.info()
#View Heatmap
import matplotlib.pyplot as plt
%matplotlib inline
df_house_corr = df_temp.corr()
fig, ax = plt.subplots(figsize=(12, 9)) 
sns.heatmap(df_house_corr, square=True, vmax=1, vmin=-1, center=0)