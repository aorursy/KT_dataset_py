import pandas as pd

import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
#index,variable name

np.array(list(zip(train.Id,train.columns)))
np.array(list(zip(train.Id,train.columns[train.isnull().any()].tolist())))
np.array(list(zip(train.Id,test.columns[test.isnull().any()].tolist())))
#train data

trainNum = train.select_dtypes(include=[np.number])

trainCat = train.select_dtypes(include=[object])

#test data

testNum = test.select_dtypes(include=[np.number])

testCat = test.select_dtypes(include=[object])
trainNum.columns[trainNum.isnull().any()].tolist()
np.array(list(zip(train.Id,trainCat.columns[trainCat.isnull().any()].tolist())))
trainNum["LotFrontage"].fillna(trainNum["LotFrontage"].mean(), inplace = True)

trainNum["MasVnrArea"].fillna(trainNum["MasVnrArea"].mean(), inplace = True)
trainNum["GarageYrBlt"].fillna(trainNum["GarageYrBlt"].value_counts().idxmax(), inplace = True)
trainCat1 = trainCat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"], axis  = 1)
trainCat1["MasVnrType"].fillna(trainCat1["MasVnrType"].value_counts().idxmax(), inplace = True)

trainCat1["BsmtCond"].fillna(trainCat1["BsmtCond"].value_counts().idxmax(), inplace = True)

trainCat1["BsmtExposure"].fillna(trainCat1["BsmtExposure"].value_counts().idxmax(), inplace = True)

trainCat1["BsmtFinType1"].fillna(trainCat1["BsmtFinType1"].value_counts().idxmax(), inplace = True)

trainCat1["BsmtFinType2"].fillna(trainCat1["BsmtFinType2"].value_counts().idxmax(), inplace = True)

trainCat1["BsmtQual"].fillna(trainCat1["BsmtQual"].value_counts().idxmax(), inplace = True)

trainCat1["Electrical"].fillna(trainCat1["Electrical"].value_counts().idxmax(), inplace = True)

trainCat1["GarageCond"].fillna(trainCat1["GarageCond"].value_counts().idxmax(), inplace = True)

trainCat1["GarageFinish"].fillna(trainCat1["GarageFinish"].value_counts().idxmax(), inplace = True)

trainCat1["GarageQual"].fillna(trainCat1["GarageQual"].value_counts().idxmax(), inplace = True)

trainCat1["GarageType"].fillna(trainCat1["GarageType"].value_counts().idxmax(), inplace = True)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
#normalizing numeric data

trainNum["MSSubClass"] = le.fit_transform(trainNum["MSSubClass"].values)

trainNum["OverallQual"] = le.fit_transform(trainNum["OverallQual"].values)

trainNum["OverallCond"] = le.fit_transform(trainNum["OverallCond"].values)

trainNum["YearBuilt"] = le.fit_transform(trainNum["YearBuilt"].values)

trainNum["YearRemodAdd"] = le.fit_transform(trainNum["YearRemodAdd"].values)

trainNum["GarageYrBlt"] = le.fit_transform(trainNum["GarageYrBlt"].values)

trainNum["YrSold"] = le.fit_transform(trainNum["YrSold"].values)
#trainCat data transformed..

trainCatTransformed = trainCat1.apply(le.fit_transform)
trainFinal = pd.concat([trainNum, trainCatTransformed], axis = 1)
trainFinal.head()
np.array(list(zip(train.Id,test.columns[test.isnull().any()].tolist())))
np.array(list(zip(train.Id,testNum.columns[testNum.isnull().any()].tolist())))
np.array(list(zip(train.Id,testCat.columns[testCat.isnull().any()].tolist())))
testNum["BsmtFinSF1"].fillna(testNum["BsmtFinSF1"].mean(), inplace = True)

testNum["BsmtFinSF2"].fillna(testNum["BsmtFinSF2"].mean(), inplace = True)

testNum["BsmtUnfSF"].fillna(testNum["BsmtUnfSF"].mean(), inplace = True)

testNum["TotalBsmtSF"].fillna(testNum["TotalBsmtSF"].mean(), inplace = True)

testNum["BsmtFullBath"].fillna(testNum["BsmtFullBath"].mean(), inplace = True)

testNum["BsmtHalfBath"].fillna(testNum["BsmtHalfBath"].mean(), inplace = True)

testNum["GarageCars"].fillna(testNum["GarageCars"].mean(), inplace = True)

testNum["GarageArea"].fillna(testNum["GarageArea"].mean(), inplace = True)

testNum["LotFrontage"].fillna(testNum["LotFrontage"].mean(), inplace = True)

testNum["MasVnrArea"].fillna(testNum["MasVnrArea"].mean(), inplace = True)

#you remember the reason for below one right?

testNum["GarageYrBlt"].fillna(testNum["GarageYrBlt"].value_counts().idxmax(), inplace = True)
# we are droping variables with percent of missing values more than 30%

testCat1 = testCat.drop(["Alley","FireplaceQu","PoolQC","Fence","MiscFeature"], axis  = 1)
#filling the NULL values with the MODE of their Variables

testCat1["MSZoning"].fillna(testCat1["MSZoning"].value_counts().idxmax(), inplace = True)

testCat1["BsmtCond"].fillna(testCat1["BsmtCond"].value_counts().idxmax(), inplace = True)

testCat1["BsmtExposure"].fillna(testCat1["BsmtExposure"].value_counts().idxmax(), inplace = True)

testCat1["BsmtFinType1"].fillna(testCat1["BsmtFinType1"].value_counts().idxmax(), inplace = True)

testCat1["BsmtFinType2"].fillna(testCat1["BsmtFinType2"].value_counts().idxmax(), inplace = True)

testCat1["BsmtQual"].fillna(testCat1["BsmtQual"].value_counts().idxmax(), inplace = True)

testCat1["Exterior1st"].fillna(testCat1["Exterior1st"].value_counts().idxmax(), inplace = True)

testCat1["GarageCond"].fillna(testCat1["GarageCond"].value_counts().idxmax(), inplace = True)

testCat1["GarageFinish"].fillna(testCat1["GarageFinish"].value_counts().idxmax(), inplace = True)

testCat1["GarageQual"].fillna(testCat1["GarageQual"].value_counts().idxmax(), inplace = True)

testCat1["GarageType"].fillna(testCat1["GarageType"].value_counts().idxmax(), inplace = True)

testCat1["Utilities"].fillna(testCat1["Utilities"].value_counts().idxmax(), inplace = True)

testCat1["Exterior2nd"].fillna(testCat1["Exterior2nd"].value_counts().idxmax(), inplace = True)

testCat1["MasVnrType"].fillna(testCat1["MasVnrType"].value_counts().idxmax(), inplace = True)

testCat1["KitchenQual"].fillna(testCat1["KitchenQual"].value_counts().idxmax(), inplace = True)

testCat1["Functional"].fillna(testCat1["Functional"].value_counts().idxmax(), inplace = True)

testCat1["SaleType"].fillna(testCat1["SaleType"].value_counts().idxmax(), inplace = True)
#normalizing numeric data

testNum["MSSubClass"] = le.fit_transform(testNum["MSSubClass"].astype(str))

testNum["OverallQual"] = le.fit_transform(testNum["OverallQual"].astype(str))

testNum["OverallCond"] = le.fit_transform(testNum["OverallCond"].astype(str))

testNum["YearBuilt"] = le.fit_transform(testNum["YearBuilt"].astype(str))

testNum["YearRemodAdd"] = le.fit_transform(testNum["YearRemodAdd"].astype(str))

testNum["GarageYrBlt"] = le.fit_transform(testNum["GarageYrBlt"].astype(str))

testNum["YrSold"] = le.fit_transform(testNum["YrSold"].astype(str))
#Transforming categorical data

testCatTransformed = testCat1.apply(le.fit_transform)
testFinal = pd.concat([testNum, testCatTransformed], axis = 1)
testFinal.head()
from sklearn import linear_model
X = trainFinal.drop(["Id","SalePrice"],axis = 1)

y = trainFinal["SalePrice"]
LR = linear_model.LinearRegression()

LR.fit(X,y)

#Liner Regression Score

LR.score(X,y)
Lasso = linear_model.Lasso(alpha=0.1)

Lasso.fit(X,y)

#Lasso Regression Score

Lasso.score(X,y)
Ridge = linear_model.Ridge(0.01)

Ridge.fit(X,y)

#Ridge Regression Score

Ridge.score(X,y)
#Linear Regression Output

submissionLR = pd.DataFrame({

        "Id":test.Id,

        "SalePrice": LR.predict(testFinal.drop("Id",axis=1))

    })



#Lasso Regression Output

submissionLasso = pd.DataFrame({

        "Id":test.Id,

        "SalePrice": Lasso.predict(testFinal.drop("Id",axis=1))

    })



#Ridge Regression Output

submissionRidge = pd.DataFrame({

        "Id":test.Id,

        "SalePrice": Ridge.predict(testFinal.drop("Id",axis=1))

    })

submissionLR.to_csv('salesPrice_LR.csv', index=False)

submissionLasso.to_csv('salesPrice_Lasso.csv', index=False)

submissionRidge.to_csv('salesPrice_Ridge.csv', index=False)