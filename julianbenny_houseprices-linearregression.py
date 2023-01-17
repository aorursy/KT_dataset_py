# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test_data = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print(train_data.shape , test_data.shape)
train_data.isnull().sum()
train_data.info()
train_data["LotFrontage"].value_counts(dropna=False)



train_data.drop("LotFrontage",axis=1,inplace=True)

test_data.drop("LotFrontage",axis=1,inplace=True)
train_data["Alley"].value_counts(dropna=False)



train_data["Alley"].fillna(value="Alley_unk",inplace=True)

test_data["Alley"].fillna(value="Alley_unk",inplace=True)
train_data["MasVnrType"].value_counts(dropna=False)



train_data["MasVnrType"].fillna(value="MaxVnrType_unk",inplace=True)

test_data["MasVnrType"].fillna(value="MaxVnrType_unk",inplace=True)
train_data["MasVnrArea"].value_counts(dropna=False)



train_data.drop("MasVnrArea",axis=1,inplace=True)

test_data.drop("MasVnrArea",axis=1,inplace=True)
train_data["BsmtQual"].value_counts(dropna=False)



train_data["BsmtQual"].fillna(value="BsmtQual_unk",inplace=True)

test_data["BsmtQual"].fillna(value="BsmtQual_unk",inplace=True)
train_data["BsmtCond"].value_counts(dropna=False)



train_data["BsmtCond"].fillna(value="BsmtCond_unk",inplace=True)

test_data["BsmtCond"].fillna(value="BsmtCond_unk",inplace=True)
train_data["BsmtExposure"].value_counts(dropna=False)



train_data["BsmtExposure"].fillna(value="BsmtExposure_unk",inplace=True)

test_data["BsmtExposure"].fillna(value="BsmtExposure_unk",inplace=True)
train_data["BsmtFinType1"].value_counts(dropna=False)



train_data["BsmtFinType1"].fillna(value="BsmtFinType1_unk",inplace=True)

test_data["BsmtFinType1"].fillna(value="BsmtFinType1_unk",inplace=True)
train_data["BsmtFinType2"].fillna(value="BsmtFinType2_unk",inplace=True)

test_data["BsmtFinType2"].fillna(value="BsmtFinType2_unk",inplace=True)
train_data["Electrical"].value_counts(dropna=False)



train_data["Electrical"].fillna(value="Electrical_unk",inplace=True)

test_data["Electrical"].fillna(value="Electrical_unk",inplace=True)
train_data["FireplaceQu"].value_counts(dropna=False)



train_data["FireplaceQu"].fillna(value="FireplaceQu_unk",inplace=True)

test_data["FireplaceQu"].fillna(value="FireplaceQu_unk",inplace=True)
train_data["GarageYrBlt"].value_counts(dropna=False)



train_data.drop("GarageYrBlt",axis=1,inplace=True)

test_data.drop("GarageYrBlt",axis=1,inplace=True)
train_data["GarageType"].fillna(value="GarageType_unk",inplace=True)

test_data["GarageType"].fillna(value="GarageType_unk",inplace=True)
train_data["GarageFinish"].fillna(value="GarageFinish_unk",inplace=True)

test_data["GarageFinish"].fillna(value="GarageFinish_unk",inplace=True)
train_data["GarageQual"].fillna(value="GarageQual_unk",inplace=True)

test_data["GarageQual"].fillna(value="GarageQual_unk",inplace=True)
train_data["GarageCond"].fillna(value="GarageCond_unk",inplace=True)

test_data["GarageCond"].fillna(value="GarageCond_unk",inplace=True)
train_data["PoolQC"].fillna(value="PoolQC_unk",inplace=True)

test_data["PoolQC"].fillna(value="PoolQC_unk",inplace=True)
train_data["Fence"].fillna(value="Fence_unk",inplace=True)

test_data["Fence"].fillna(value="Fence_unk",inplace=True)
train_data["MiscFeature"].fillna(value="MiscFeature_unk",inplace=True)

test_data["MiscFeature"].fillna(value="MiscFeature_unk",inplace=True)
cols_with_missing = [col for col in test_data.columns 

                                 if test_data[col].isnull().any()]

test_data = test_data.drop(cols_with_missing, axis=1)

train_data = train_data.drop(cols_with_missing, axis=1)



(train_data.dtypes != object).sum()
not_categorical = train_data.drop(["Id","SalePrice"],axis=1).columns[train_data.drop(["Id","SalePrice"],axis=1).dtypes != object]

not_categorical
for i in not_categorical:

    ran = train_data[i].max(axis=0) - train_data[i].min(axis=0)

    avg = train_data[i].mean(axis=0)

    train_data[i] = (train_data[i]-avg)/ran



train_data[not_categorical].head()
for i in not_categorical:

    ran = test_data[i].max(axis=0) - test_data[i].min(axis=0)

    avg = test_data[i].mean(axis=0)

    test_data[i] = (test_data[i]-avg)/ran



test_data[not_categorical].head()
print(train_data.shape , test_data.shape)
categorical = train_data.drop(["Id","SalePrice"],axis=1).columns[train_data.drop(["Id","SalePrice"],axis=1).dtypes == object]

categorical
pd.get_dummies(train_data[categorical[2]])
for i in categorical:

    dummies = pd.get_dummies(train_data[i])

    dummies.columns = str(i) +"_"+ dummies.columns

    train_data = pd.concat([train_data,dummies],axis=1)

    train_data.drop(i,axis=1,inplace=True)
for i in categorical:

    dummies = pd.get_dummies(test_data[i])

    dummies.columns = str(i) +"_"+ dummies.columns

    test_data = pd.concat([test_data,dummies],axis=1)

    test_data.drop(i,axis=1,inplace=True)
print(train_data.shape , test_data.shape)
for i in train_data.drop("SalePrice",axis=1).columns:

    if i not in test_data.columns:

        test_data[i] = np.zeros(len(test_data))
for i in test_data.columns:

    if i not in train_data.columns:

        train_data[i] = np.zeros(len(train_data))
print(train_data.shape , test_data.shape)
train_data.keys()
test_data = test_data[train_data.drop("SalePrice",axis=1).columns]
test_data.keys()
X = train_data.drop(["Id","SalePrice"],axis=1)

y = train_data["SalePrice"]



from sklearn.model_selection import train_test_split



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.05)
print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
from sklearn.linear_model import LinearRegression



reg = LinearRegression()

reg.fit(X_train,y_train)

reg.score(X_test,y_test)
pred = reg.predict(test_data.drop("Id",axis=1))
pred.shape
submission = pd.DataFrame(data=pred,columns=["SalePrice"])
submission["Id"] = test_data["Id"]
submission.set_index("Id",inplace=True)
submission.shape
submission.to_csv("submission.csv")