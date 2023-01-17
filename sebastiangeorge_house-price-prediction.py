# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

train.head()


#index,variable name

np.array(list(zip(train.Id,train.columns)))
#attributes which have null in train 

np.array(list(zip(train.Id,train.columns[train.isnull().any()].tolist())))



#which all varriables in test has null values 

np.array(list(zip(train.Id,test.columns[test.isnull().any()].tolist())))

#train data

trainNum = train.select_dtypes(include=[np.number])

trainCat = train.select_dtypes(include=[object])

#test data

testNum = test.select_dtypes(include=[np.number])

testCat = test.select_dtypes(include=[object])
trainNum.head()
#which all values in trainNum has missing values 

trainNum.columns[trainNum.isnull().any()].tolist()
#What variables in trainCat have NULL values in it?





np.array(list(zip(train.Id,trainCat.columns[trainCat.isnull().any()].tolist())))





#Handling NULL values in Numerical Variables



trainNum["LotFrontage"].fillna(trainNum["LotFrontage"].mean(), inplace = True)

trainNum["MasVnrArea"].fillna(trainNum["MasVnrArea"].mean(), inplace = True)
trainNum["GarageYrBlt"].fillna(trainNum["GarageYrBlt"].value_counts().idxmax(), inplace = True)


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
# handling missing values in catagorical 

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
#encoding 

#since all missing values are filled 

#trainCat data transformed..

trainCatTransformed = trainCat1.apply(le.fit_transform)
trainFinal = pd.concat([trainNum, trainCatTransformed], axis = 1)

np.array(list(zip(train.Id,test.columns[test.isnull().any()].tolist())))
np.array(list(zip(train.Id,testNum.columns[testNum.isnull().any()].tolist())))



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
from sklearn import linear_model
X = trainFinal.drop(["Id","SalePrice"],axis = 1)

y = trainFinal["SalePrice"]
LR = linear_model.LinearRegression()

LR.fit(X,y)

#Liner Regression Score

LR.score(X,y)
LR.predict(testFinal.drop("Id",axis=1))
#Linear Regression Output

submissionLR = pd.DataFrame({

        "Id":test.Id,

        "SalePrice": LR.predict(testFinal.drop("Id",axis=1))

    })
submissionLR.to_csv('salesPrice_LR.csv', index=False)