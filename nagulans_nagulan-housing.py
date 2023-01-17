# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.drop(labels=["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"],axis=1,inplace=True)
test.drop(labels=["Alley","PoolQC","Fence","MiscFeature","FireplaceQu"],axis=1,inplace=True)
train.dropna(inplace=True,axis=0,subset=["Electrical","MasVnrType"])
#test.dropna(inplace=True,axis=0,subset=["Exterior1st","Exterior2nd","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF","TotalBsmtSF","KitchenQual","GarageCars","GarageArea","SaleType","BsmtFullBath","BsmtHalfBath"])
test
train["LotFrontage"].fillna(train["LotFrontage"].mean(),inplace=True)
train["MasVnrArea"].fillna(train["MasVnrArea"].mean(),inplace=True)
train["GarageYrBlt"].fillna(train["GarageYrBlt"].mean(),inplace=True)
train["BsmtCond"].fillna("TA",inplace=True)
train["BsmtQual"].fillna("TA",inplace=True)
train["BsmtExposure"].fillna("No",inplace=True)
train["BsmtFinType1"].fillna("Unf",inplace=True)
train["BsmtFinType2"].fillna("Unf",inplace=True)
train["GarageFinish"].fillna("Attchd",inplace=True)
train["GarageType"].fillna("Unf",inplace=True)
train["GarageQual"].fillna("TA",inplace=True)
train["GarageCond"].fillna("TA",inplace=True)

test["LotFrontage"].fillna(test["LotFrontage"].mean(),inplace=True)
test["MasVnrArea"].fillna(test["MasVnrArea"].mean(),inplace=True)
test["GarageYrBlt"].fillna(test["GarageYrBlt"].mean(),inplace=True)
test["BsmtFinSF2"].fillna(test["BsmtFinSF2"].mean(),inplace=True)
test["BsmtFinSF1"].fillna(test["BsmtFinSF1"].mean(),inplace=True)
test["BsmtUnfSF"].fillna(test["BsmtUnfSF"].mean(),inplace=True)
test["TotalBsmtSF"].fillna(test["TotalBsmtSF"].mean(),inplace=True)
test["BsmtFullBath"].fillna(test["BsmtFullBath"].mean(),inplace=True)
test["BsmtHalfBath"].fillna(test["BsmtHalfBath"].mean(),inplace=True)
test["GarageArea"].fillna(test["GarageArea"].mean(),inplace=True)
test["GarageCars"].fillna(test["GarageCars"].mean(),inplace=True)
test["Functional"].fillna("Typ",inplace=True)
test["Utilities"].fillna("AllPub",inplace=True)
test["MSZoning"].fillna("RL",inplace=True)
test["BsmtCond"].fillna("TA",inplace=True)
test["BsmtFinType2"].fillna("Unf",inplace=True)
test["GarageQual"].fillna("TA",inplace=True)
test["GarageCond"].fillna("TA",inplace=True)
test["GarageFinish"].fillna("Attchd",inplace=True)
test["GarageType"].fillna("Unf",inplace=True)
test["BsmtFinType1"].fillna("Unf",inplace=True)
test["BsmtQual"].fillna("TA",inplace=True)
test["BsmtExposure"].fillna("No",inplace=True)
test["MasVnrType"].fillna("None",inplace=True)
test["Exterior1st"].fillna("VinylSd",inplace=True)
test["Exterior2nd"].fillna("VinylSd",inplace=True)
test["KitchenQual"].fillna("TA",inplace=True)
test["MasVnrType"].fillna("None",inplace=True)
test["MasVnrType"].fillna("None",inplace=True)
test["SaleType"].fillna("WD",inplace=True)

null_columns=test.columns[test.isnull().any()]
print(test[null_columns].isnull().sum())
for col in null_columns:
    print(col,test[col].unique(),test[col].value_counts())
print(train.dtypes)
main_column=[]
drop_columns=[]
t=train
for i in t:
    if(t[i].dtype == object):
        drop_columns.append(i)
        train=pd.concat([train,pd.get_dummies(train[i],prefix=i)],axis=1)
        test=pd.concat([test,pd.get_dummies(test[i],prefix=i)],axis=1)
train.drop(labels=drop_columns,axis=1,inplace=True)
#temp=train["SalePrice"]
#train.drop(labels="SalePrice",axis=1,inplace=True)
#train=pd.concat(temp,axis=1)
test.drop(labels=drop_columns,axis=1,inplace=True)
for i in train:
    if(i!="SalePrice" and train[i].dtype != object and abs(train[i].corr(train["SalePrice"]))>0.3):
        print(i)
        main_column.append(i)
main_column.append("SalePrice")
print(len(main_column))
plt.figure(figsize=(30,20))
sns.heatmap(train[main_column].corr(),cmap="bwr",linewidth=0.5,vmin=-1,annot=True)
main_column.pop()
print(main_column)
x=train[main_column]
y=train["SalePrice"]
#from sklearn import linear_model
#regr = linear_model.LinearRegression()
from sklearn.ensemble import RandomForestRegressor
regr= RandomForestRegressor(max_depth=50,random_state=0)
regr.fit(x, y)
price = regr.predict(test[main_column])
print(len(price))
index=[]
for i in range(1461,2920):
    index.append(i)
index= np.array(index)
dataset = pd.DataFrame({'Id': index, 'SalePrice':price}, columns=['Id', 'SalePrice'])
#axl=sns.distplot(test["SalePrice"],hist=False,color="r")
#sns.distplot(price,hist=False,color="b",ax=axl)
dataset.to_csv('submission.csv', index=False)
print(dataset)