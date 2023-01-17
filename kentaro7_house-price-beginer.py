import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

pd.set_option('display.max_rows', None)

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns 

import random

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("."))

['gender_submission.csv','train.csv','test.csv']
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
print(train.info())

print('*'*30)

print(test.info())
train.drop(1298,axis=0,inplace=True)

train.drop(523,axis=0,inplace=True)

train["SalePrice"] = train.SalePrice.apply(np.log)
sale_price = train.SalePrice
all_data = pd.concat([train,test],axis=0,ignore_index=False)

all_data.drop("SalePrice",axis=1,inplace=True)
print(all_data.isnull().sum().sort_values(ascending=False))
print("*"*30)

print("PoolQC")

print('*'*30)

all_data.PoolQC.fillna("N",inplace=True)

print(all_data.PoolQC.value_counts())
print("*"*30)

print("MiscFeature")  

print("*"*30)

print(all_data.MiscFeature.value_counts())

all_data.MiscFeature.fillna("N",inplace=True)
print("*"*30)

print("Alley")

print("*"*30)

print(all_data.Alley.value_counts())

all_data.Alley.fillna("N",inplace=True)
print("*"*30)

print("Fence")

print("*"*30)

print(all_data.Fence.value_counts())

all_data.Fence.fillna("N",inplace=True)
print("*"*30)

print("FireplaceQu")

print("*"*30)

print(all_data.FireplaceQu.value_counts())

all_data.FireplaceQu.fillna("N",inplace=True)
print("*"*30)

print("LotFrontage")

print("*"*30)

all_data.LotFrontage.fillna(0,inplace=True)

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x:x.fillna(x.median()))
print("*"*30)

print("Garage")

print("*"*30)

print("Cond")

print(all_data.GarageCond.value_counts())

print("Finish")

print(all_data.GarageFinish.value_counts())

print("Qual")

print(all_data.GarageQual.value_counts())

print("Yrbuil")

print("mean : "+str(all_data.GarageYrBlt.mean()))
all_data.GarageFinish[all_data["Id"]==2577].fillna("Unf",inplace=True)

all_data.GarageCond[all_data["Id"]==2577].fillna("TA",inplace=True)

all_data.GarageQual[all_data["Id"]==2577].fillna("TA",inplace=True)

all_data.GarageYrBlt[all_data["Id"]==2577].fillna(1978,inplace=True)



all_data.GarageFinish[all_data["Id"]==2127].fillna("Fin",inplace=True)

all_data.GarageCond[all_data["Id"]==2127].fillna("TA",inplace=True)

all_data.GarageQual[all_data["Id"]==2127].fillna("TA",inplace=True)

all_data.GarageYrBlt[all_data["Id"]==2127].fillna(1978,inplace=True)



all_data.GarageFinish.fillna("N",inplace=True)

all_data.GarageCond.fillna("N",inplace=True)

all_data.GarageQual.fillna("N",inplace=True)

all_data.GarageYrBlt.fillna("0",inplace=True)

all_data.GarageType.fillna("N",inplace=True)
all_data.GarageCars.fillna(1,inplace=True)

all_data.GarageArea.fillna(0,inplace=True)
all_data.drop("GarageArea",inplace=True,axis=1)
print("*"*30)

print("Bsmt")

print("*"*30)

all_data.BsmtCond.fillna("N",inplace=True)

all_data.BsmtExposure.fillna("N",inplace=True)

all_data.BsmtFinType2.fillna("N",inplace=True)

all_data.BsmtFinType1.fillna("N",inplace=True)

all_data.BsmtQual.fillna("N",inplace=True)

all_data.BsmtFinSF1.fillna(0,inplace=True)

all_data.BsmtFinSF2.fillna(0,inplace=True)

all_data.BsmtFullBath.fillna(0,inplace=True)

all_data.BsmtHalfBath.fillna(0,inplace=True)

all_data.BsmtUnfSF.fillna(0,inplace=True)

all_data.TotalBsmtSF.fillna(0,inplace=True)
print("*"*30)

print("MasVnr")

print("*"*30)

all_data.MasVnrType.fillna("None",inplace=True)

all_data.MasVnrArea.fillna(0,inplace=True)
print("*"*30)

print("MsZoning")

print("*"*30)

print(all_data.MSSubClass[all_data["MSZoning"].isnull()==True])

all_data["MSZoning"] = all_data["MSZoning"].fillna(all_data["MSZoning"].mode()[0])
print("*"*30)

print("Functional")

print("*"*30)

print(all_data.Functional.value_counts())

all_data.Functional.fillna("Typ",inplace=True)
print("*"*30)

print("The other")

print("*"*30)



print(all_data.Exterior1st.value_counts())

print(all_data.SaleType.value_counts())

print(all_data.Electrical.value_counts())

print(all_data.Exterior2nd.value_counts())

print(all_data.KitchenQual.value_counts())

all_data["Exterior1st"] = all_data["Exterior1st"].fillna(all_data["Exterior1st"].mode()[0])

all_data["Exterior2nd"] = all_data["Exterior2nd"].fillna(all_data["Exterior2nd"].mode()[0])

all_data["SaleType"] = all_data["SaleType"].fillna(all_data["SaleType"].mode()[0])

all_data["Electrical"] = all_data["Electrical"].fillna(all_data["Electrical"].mode()[0])

all_data["KitchenQual"] = all_data["KitchenQual"].fillna(all_data["KitchenQual"].mode()[0])
all_data = pd.get_dummies(all_data)
print("*"*30)

print("Study")

print("*"*30)

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error
Id = test.Id

all_data = all_data.drop("Id",axis=1)

df_train = all_data[:1458]

df_test = all_data[1458:]

 

X = df_train

y = sale_price



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)



lir = LinearRegression()

lir.fit(X_train,y_train)

R = lir.predict(X_test)

print(np.sqrt(mean_squared_error(R,y_test)))
print("*"*30)

print("submit")

print("*"*30)

test_predict = lir.predict(df_test)

for i in range(1459):

    if test_predict[i] <= 0:

        test_predict[i] = 11.0

    elif test_predict[i] >=15.0:

        test_predict[i] = 12.0

test_predict = np.exp(test_predict)

data = {"Id" : Id,"SalePrice":test_predict}

submission = pd.DataFrame(data=data,dtype=int)

submission.Id = submission.Id.astype(int)

submission.SalePrice = submission.SalePrice.astype(int)

submission.to_csv("house_submission.csv",index =False)