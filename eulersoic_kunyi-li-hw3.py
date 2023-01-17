import seaborn as sns

import matplotlib.pyplot as plt
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
ss = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.info()
test.isnull().sum()
train = train.drop(train[["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]],axis=1)

test = test.drop(test[["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"]],axis=1)

train
train["LotFrontage"].value_counts().plot.bar()

plt.show()
train["LotFrontage"].describe()
train["LotFrontage"].fillna(train["LotFrontage"].mean(),inplace=True)
test["LotFrontage"].value_counts().plot.bar()

plt.show()
test["LotFrontage"].fillna(test["LotFrontage"].mean(),inplace=True)
train.info()
plt.subplots(figsize=(20,20))

ax = plt.axes()

corr = train.corr()

sns.heatmap(corr)
train1 = train[["SalePrice","OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]]

train1.head()
test1 = test[["OverallQual","YearBuilt","YearRemodAdd","MasVnrArea","TotalBsmtSF","1stFlrSF","GrLivArea","FullBath","TotRmsAbvGrd","GarageCars","GarageArea"]]

test1.head()
train1["MasVnrType"] = train["MasVnrType"]

test1["MasVnrType"] = train["MasVnrType"]
train1["ExterQual"] = train["ExterQual"]

test1["ExterQual"] = test["ExterQual"]
train1["Foundation"] = train["Foundation"]

test1["Foundation"] = test["Foundation"]
train1["GarageType"] = train["GarageType"]

test1["GarageType"] = test["GarageType"]
train1["GarageFinish"] = train["GarageFinish"]

test1["GarageFinish"] = test["GarageFinish"]
train1.info()
train["MasVnrType"].value_counts().plot.bar()

plt.show()
train1["MasVnrType"] = train1["MasVnrType"].map({"None":1,"BrkFace":2,"Stone":3,"BrkCmn":4})

train1["MasVnrType"].fillna(method='ffill',inplace = True)

train1["MasVnrType"].value_counts().plot.bar()

plt.show()
train1["MasVnrArea"].value_counts().plot.bar()

plt.show()
train1["MasVnrArea"].describe()
train1["MasVnrArea"].fillna(method='ffill',inplace = True)
train1["GarageType"].value_counts().plot.bar()

plt.show()
train1["GarageType"] = train1["GarageType"].map({"Attchd":1,"Detchd":2,"BuiltIn":3,"Basment":4,"CarPort":5,"2Types":6})

train1["GarageType"].fillna(method='ffill',inplace = True)
train1["GarageType"].value_counts().plot.bar()

plt.show()
train1["GarageFinish"].value_counts().plot.bar()

plt.show()
train1["GarageFinish"] = train1["GarageFinish"].map({"Unf":1,"RFn":2,"Fin":3})

train1["GarageFinish"].fillna(method='ffill',inplace = True)

train1["GarageFinish"].value_counts().plot.bar()

plt.show()
train1["ExterQual"].value_counts().plot.bar()

plt.show()
train1["ExterQual"] = train1["ExterQual"].map({"TA":1,"Gd":2,"Ex":3,"Fa":4})

train1["ExterQual"].value_counts().plot.bar()

plt.show()
train1["Foundation"].value_counts().plot.bar()

plt.show()
train1["Foundation"] = train1["Foundation"].map({"PConc":1,"CBlock":2,"BrkTil":3,"Slab":4,"Stone":5,"Wood":6})
train1["Foundation"].value_counts().plot.bar()

plt.show()
train1.info()
plt.subplots(figsize=(20,20))

ax = plt.axes()

corr = train1.corr()

sns.heatmap(corr)
traindata = train1.drop(train[["MasVnrType","Foundation","GarageType","GarageFinish"]],axis=1)

testdata = test1.drop(train[["MasVnrType","Foundation","GarageType","GarageFinish"]],axis=1)
traindata.head(10)
testdata.info()
testdata.head(10)
testdata["MasVnrArea"].fillna(method='ffill',inplace=True)
testdata.fillna(testdata.mean(),inplace=True)
testdata["ExterQual"].value_counts().plot.bar()

plt.show()
testdata["ExterQual"] = testdata["ExterQual"].map({"TA":1,"Gd":2,"Ex":3,"Fa":4})

testdata["ExterQual"].value_counts().plot.bar()

plt.show()
testdata.info()
x_train = traindata.iloc[:,1:]

y_train = traindata.iloc[:,0]

x_test = testdata

y_test = ss.iloc[:,1]
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier()

clf.fit(x_train,y_train)
clf.score(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=0).fit(x_train, y_train)



#Returns the coefficient of determination R^2 of the prediction.

model.score(x_train, y_train)
model.score(x_test, y_test)