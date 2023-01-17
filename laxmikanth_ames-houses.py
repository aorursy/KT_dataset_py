import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
houseprice=pd.read_csv("../input/train.csv")
houseprice.isnull().sum()
housedfnum=houseprice.select_dtypes(include=[np.number])
housedfcat=houseprice.select_dtypes(include=[object])
housedfnum["LotFrontage"].fillna(housedfnum["LotFrontage"].mean(),inplace=True)

housedfnum["MasVnrArea"].fillna(housedfnum["MasVnrArea"].mean(),inplace=True)

housedfnum["GarageYrBlt"].fillna(housedfnum["GarageYrBlt"].value_counts().idxmax(),inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
housedfnum["MSSubClass"]=le.fit_transform(housedfnum["MSSubClass"].values)

housedfnum["OverallQual"]=le.fit_transform(housedfnum["OverallQual"].values)

housedfnum["OverallCond"]=le.fit_transform(housedfnum["OverallCond"].values)

housedfnum["YearBuilt"]=le.fit_transform(housedfnum["YearBuilt"].values)

housedfnum["YearRemodAdd"]=le.fit_transform(housedfnum["YearRemodAdd"].values)

housedfnum["YrSold"]=le.fit_transform(housedfnum["YrSold"].values)

housedfnum["GarageYrBlt"]=le.fit_transform(housedfnum["GarageYrBlt"].values)
housedfcat1=housedfcat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)
housedfcat1["MasVnrType"].fillna(housedfcat1["MasVnrType"].value_counts().idxmax(),inplace=True)

housedfcat1["BsmtQual"].fillna(housedfcat1["BsmtQual"].value_counts().idxmax(),inplace=True)

housedfcat1["BsmtCond"].fillna(housedfcat1["BsmtCond"].value_counts().idxmax(),inplace=True)

housedfcat1["BsmtExposure"].fillna(housedfcat1["BsmtExposure"].value_counts().idxmax(),inplace=True)

housedfcat1["BsmtFinType1"].fillna(housedfcat1["BsmtFinType1"].value_counts().idxmax(),inplace=True)

housedfcat1["BsmtFinType2"].fillna(housedfcat1["BsmtFinType2"].value_counts().idxmax(),inplace=True)

housedfcat1["Electrical"].fillna(housedfcat1["Electrical"].value_counts().idxmax(),inplace=True)

housedfcat1["GarageType"].fillna(housedfcat1["GarageType"].value_counts().idxmax(),inplace=True)

housedfcat1["GarageFinish"].fillna(housedfcat1["GarageFinish"].value_counts().idxmax(),inplace=True)

housedfcat1["GarageQual"].fillna(housedfcat1["GarageQual"].value_counts().idxmax(),inplace=True)

housedfcat1["GarageCond"].fillna(housedfcat1["GarageCond"].value_counts().idxmax(),inplace=True)
housedfcat2=housedfcat1.apply(le.fit_transform)
housefinal=pd.concat([housedfnum,housedfcat2],axis=1)
housefinal.shape
from sklearn.linear_model import LinearRegression
LiR=LinearRegression()
y=housefinal["SalePrice"]

X=housefinal.drop(["Id","SalePrice"],axis=1)
LiR.fit(X,y)
LiR.score(X,y)
from sklearn.model_selection import cross_val_score
lircross=cross_val_score(LiR,X,y,cv=10)
lircross
print("Accuracy: %0.2f(+/- %0.2f)" % (lircross.mean(), lircross.std()*2))
predictedprice=LiR.predict(X)
predictedprice
priceresidual=housefinal.SalePrice-predictedprice
np.sqrt(np.mean((priceresidual)**2))  
housetest=pd.read_csv("../input/test.csv")
housetest.isnull().sum()
housetestnum=housetest.select_dtypes(include=[np.number])
housetestcat=housetest.select_dtypes(include=[object])
housetestnum.isnull().sum()
housetestnum["LotFrontage"].fillna(housetestnum["LotFrontage"].mean(),inplace=True)

housetestnum["MasVnrArea"].fillna(housetestnum["MasVnrArea"].mean(),inplace=True)

housetestnum["BsmtFinSF1"].fillna(housetestnum["BsmtFinSF1"].mean(),inplace=True)

housetestnum["BsmtFinSF2"].fillna(housetestnum["BsmtFinSF2"].mean(),inplace=True)

housetestnum["BsmtUnfSF"].fillna(housetestnum["BsmtUnfSF"].mean(),inplace=True)

housetestnum["TotalBsmtSF"].fillna(housetestnum["TotalBsmtSF"].mean(),inplace=True)

housetestnum["BsmtFullBath"].fillna(housetestnum["BsmtFullBath"].mean(),inplace=True)

housetestnum["BsmtHalfBath"].fillna(housetestnum["BsmtHalfBath"].mean(),inplace=True)

housetestnum["GarageCars"].fillna(housetestnum["GarageCars"].mean(),inplace=True)

housetestnum["GarageArea"].fillna(housetestnum["GarageArea"].mean(),inplace=True)

housetestnum["GarageYrBlt"].fillna(housetestnum["GarageYrBlt"].value_counts().idxmax(),inplace=True)
housetestnum.isnull().sum()
housetestcat.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
housetestnum["MSSubClass"]=le.fit_transform(housetestnum["MSSubClass"].values)

housetestnum["OverallQual"]=le.fit_transform(housetestnum["OverallQual"].values)

housetestnum["OverallCond"]=le.fit_transform(housetestnum["OverallCond"].values)

housetestnum["YearBuilt"]=le.fit_transform(housetestnum["YearBuilt"].values)

housetestnum["YearRemodAdd"]=le.fit_transform(housetestnum["YearRemodAdd"].values)

housetestnum["YrSold"]=le.fit_transform(housetestnum["YrSold"].values)

housetestnum["GarageYrBlt"]=le.fit_transform(housetestnum["GarageYrBlt"].values)
housetestcat1=housetestcat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"],axis=1)
housetestcat1.isnull().sum()
housetestcat1["MasVnrType"].fillna(housetestcat1["MasVnrType"].value_counts().idxmax(),inplace=True)

housetestcat1["BsmtQual"].fillna(housetestcat1["BsmtQual"].value_counts().idxmax(),inplace=True)

housetestcat1["BsmtCond"].fillna(housetestcat1["BsmtCond"].value_counts().idxmax(),inplace=True)

housetestcat1["BsmtExposure"].fillna(housetestcat1["BsmtExposure"].value_counts().idxmax(),inplace=True)

housetestcat1["BsmtFinType1"].fillna(housetestcat1["BsmtFinType1"].value_counts().idxmax(),inplace=True)

housetestcat1["BsmtFinType2"].fillna(housetestcat1["BsmtFinType2"].value_counts().idxmax(),inplace=True)

housetestcat1["GarageType"].fillna(housetestcat1["GarageType"].value_counts().idxmax(),inplace=True)

housetestcat1["GarageFinish"].fillna(housetestcat1["GarageFinish"].value_counts().idxmax(),inplace=True)

housetestcat1["GarageQual"].fillna(housetestcat1["GarageQual"].value_counts().idxmax(),inplace=True)

housetestcat1["GarageCond"].fillna(housetestcat1["GarageCond"].value_counts().idxmax(),inplace=True)
housetestcat1["MSZoning"]=le.fit_transform(housetestcat1["MSZoning"].astype(str))
housetestcat1["Utilities"]=le.fit_transform(housetestcat1["Utilities"].astype(str))
housetestcat1["Exterior1st"]=le.fit_transform(housetestcat1["Exterior1st"].astype(str))
housetestcat1["Exterior2nd"]=le.fit_transform(housetestcat1["Exterior2nd"].astype(str))
housetestcat1["KitchenQual"]=le.fit_transform(housetestcat1["KitchenQual"].astype(str))
housetestcat1["Functional"]=le.fit_transform(housetestcat1["Functional"].astype(str))
housetestcat1["SaleType"]=le.fit_transform(housetestcat1["SaleType"].astype(str))
housetestcat1.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
housetestcat2=housetestcat1.apply(le.fit_transform)
housetestcat2.isnull().sum()
housetestfinal=pd.concat([housetestnum,housetestcat2],axis=1)
housetestfinal.shape
housetestfinal1=housetestfinal.drop(["Id"],axis=1)
from sklearn.linear_model import LinearRegression
LiR=LinearRegression()
housetestfinal1.head()
LiR.fit(X,y)
salespredict=LiR.predict(housetestfinal1)
salespredict
np.sqrt(np.mean((salespredict)**2))
testpredict=pd.DataFrame(salespredict)
testpredict.describe()