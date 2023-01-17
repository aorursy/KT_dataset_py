import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train=pd.read_csv("../input/train.csv")
train.shape
train.describe()
train.isnull().sum()
plt.hist(train["SalePrice"],bins=50)
plt.hist(np.log(train["SalePrice"]),bins=50)
plt.scatter(train["YearBuilt"],train["SalePrice"])
y1=np.log(train["SalePrice"])
plt.scatter(train["YearBuilt"],y1)
numerical=list(train.describe().columns)
plt.figure(figsize=(20,10))
for i in numerical:
    if (i!="SalePrice"):
        plt.scatter(train[i],train["SalePrice"],label=i)
        plt.legend(loc=4)
e=plt.boxplot(train["SalePrice"])

iqr=train["SalePrice"].quantile(0.75)-train["SalePrice"].quantile(0.25)
iqr1=train["SalePrice"].quantile(0.25)-1.5*iqr
iqr2=train["SalePrice"].quantile(0.75)+1.5*iqr
rem_outlier=(train["SalePrice"] <= iqr2) & (train["SalePrice"] >= iqr1)
train2=train[rem_outlier]
y=np.log(train2["SalePrice"])
plt.scatter(train2["YearBuilt"],y)
p=np.polyfit(train2["YearBuilt"],train2["SalePrice"],1)


c=np.poly1d(p)

plt.scatter(train2["YearBuilt"],train2["SalePrice"])
plt.scatter(train2["YearBuilt"],c(train2["YearBuilt"]))
plt.scatter(train2["YearBuilt"],-(c(train2["YearBuilt"])-train2["SalePrice"]))
p1=np.polyfit(train2["GrLivArea"],train2["SalePrice"],3)
c1=np.poly1d(p1)
plt.scatter(train2["GrLivArea"],train2["SalePrice"])
plt.scatter(train2["GrLivArea"],c1(train2["GrLivArea"]))
plt.scatter(train2["GrLivArea"],-(c1(train2["GrLivArea"])-train2["SalePrice"]))
p=np.polyfit(train2["TotalBsmtSF"][(train2["TotalBsmtSF"]!=0) & (train2["TotalBsmtSF"]<=3000)],train2["SalePrice"][(train2["TotalBsmtSF"]!=0) & (train2["TotalBsmtSF"]<=3000)],3)
c=np.poly1d(p)
plt.scatter(train2["TotalBsmtSF"][train2["TotalBsmtSF"]!=0],train2["SalePrice"][train2["TotalBsmtSF"]!=0])
plt.scatter(train2["TotalBsmtSF"][(train2["TotalBsmtSF"]!=0) & (train2["TotalBsmtSF"]<=3000)],c(train2["TotalBsmtSF"][(train2["TotalBsmtSF"]!=0) & (train2["TotalBsmtSF"]<=3000)]))
sns.boxplot(x=train2["MSSubClass"],y=train2["SalePrice"])
sns.boxplot(x=train2["LotShape"],y=train2["SalePrice"])
sns.boxplot(x=train2["LotConfig"],y=train2["SalePrice"])
plt.figure(figsize=(20,10))
sns.boxplot(x=train2["Neighborhood"],y=train2["SalePrice"])
sns.boxplot(x=train2["Condition1"],y=train2["SalePrice"])
sns.boxplot(x=train2["HouseStyle"],y=train2["SalePrice"])
sns.boxplot(x=train2["OverallQual"],y=train2["SalePrice"])
sns.boxplot(x=train2["OverallCond"],y=train2["SalePrice"])
train3=train2.select_dtypes(exclude="object")
var=train3.columns.tolist()
var.remove("SalePrice")
var.remove("Id")
train3=train3.dropna()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(train3[var],train3["SalePrice"],random_state=0)
xtrain.shape
from sklearn.linear_model import LinearRegression
lr1=LinearRegression().fit(xtrain,ytrain)
lr1.score(xtrain,ytrain)
lr1.score(xtest,ytest)
from sklearn.linear_model import Ridge
rid=Ridge(alpha=120).fit(xtrain,ytrain)
rid.score(xtrain,ytrain)
rid.score(xtest,ytest)
from sklearn.linear_model import Lasso
las=Lasso(alpha=120).fit(xtrain,ytrain)
las.score(xtrain,ytrain)
las.score(xtest,ytest)
las.coef_
test=pd.read_csv("../input/test.csv")
test1=test.select_dtypes(exclude="object")
var=test1.columns.tolist()
var.remove("Id")
test2=test1[var].dropna()
testpred=las.predict(test2)
plt.hist(testpred)