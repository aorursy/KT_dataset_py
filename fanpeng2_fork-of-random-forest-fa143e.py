import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn as sk

import sklearn.preprocessing

import sklearn.ensemble

import sklearn.model_selection

#import tensorflow as tf

np.random.seed(0)
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

all_data = pd.concat([train, test], ignore_index=True)

all_data.drop(["Id", "SalePrice"], axis=1, inplace=True)
df = all_data.copy()

df.MSSubClass = df.MSSubClass.apply(str) #MSSubClass: Identifies the type of dwelling involved in the sale.

df.Alley = df.Alley.fillna("NA") #Alley: Type of alley access to property

df.LotShape = df.LotShape.astype("category", categories=["Reg","IR1","IR2","IR3"])

df.Utilities = df.Utilities.astype("category", categories=["AllPub","NoSewr","NoSeWa","ELO"])

df.LandSlope = df.LandSlope.astype("category", categories=["Gtl","Mod","Sev"])

df.ExterQual = df.ExterQual.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.ExterCond = df.ExterCond.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.BsmtQual = df.BsmtQual.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.BsmtCond = df.BsmtCond.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.BsmtExposure = df.BsmtExposure.astype("category", categories=["Gd","Av","Mn","No"])

df.BsmtFinType1 = df.BsmtFinType1.astype("category", categories=["GLQ","ALQ","BLQ","Rec","LwQ","Unf"])

df.BsmtFinType2 = df.BsmtFinType2.astype("category", categories=["GLQ","ALQ","BLQ","Rec","LwQ","Unf"])

df.HeatingQC = df.HeatingQC.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.CentralAir = df.CentralAir.astype("category", categories=["N","Y"])

df.KitchenQual = df.KitchenQual.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.Functional = df.Functional.astype("category", categories=["Typ","Min1","Min2","Mod","Maj1","Maj2","Sev","Sal"])

df.FireplaceQu = df.FireplaceQu.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.GarageFinish = df.GarageFinish.astype("category", categories=["Fin","RFn","Unf"])

df.GarageQual = df.GarageQual.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.GarageCond = df.GarageCond.astype("category", categories=["Ex","Gd","TA","Fa","Po"])

df.PavedDrive = df.PavedDrive.astype("category", categories=["Y","P","N"])

df.PoolQC = df.PoolQC.astype("category", categories=["Ex","Gd","TA","Fa"])

df.Fence = df.Fence.astype("category", categories=["GdPrv","MnPrv","GdWo","MnWw"])
#df = all_data
pd.set_option('display.max_rows', df.shape[1])

print(df.describe(percentiles=[]).transpose())
#all_data.dtypes

#all_data.head(30)

df1 = df.select_dtypes(include=['object'])

#for i, col in enumerate(categorical_data.columns):

#    plt.figure(i)

#    sns.countplot(x=col, data=categorical_data)

for col in df1.columns:

    if not df1[col].isnull().values.any():

        continue

    print(df1[col].value_counts(dropna=False))

    print()
for col in df.columns:

    if df[col].dtype == 'object':

        continue

    if not df[col].isnull().values.any():

        continue    

    df[col + '_NA'] = np.where(pd.isnull(df[col]), 1, 0)

for col in df.columns:

    if df[col].dtype.name == 'category':

        df[col] = df[col].fillna(df[col].mode().iloc[0])

        df[col] = df[col].cat.codes
df = pd.get_dummies(df)

df = df.fillna(df.mean())
#df.dtypes

#df.head(30)

#df.BsmtQual.mode()

#df.PoolQC
x = df[:train.shape[0]]

y = train.SalePrice

x_test = df[train.shape[0]:]
y = np.log(y)
def submit(filename):

    submission = pd.DataFrame({"Id": test.Id, "SalePrice": np.exp(y_test)})

    submission.to_csv(filename + ".csv", index = False)
model_gb = sk.ensemble.GradientBoostingRegressor()

model_gb.fit(x, y)

scores = np.sqrt(-sk.model_selection.cross_val_score(model_gb, x, y, cv=5, scoring="neg_mean_squared_error"))

print(scores)

print(scores.mean(), scores.std())

y_test = model_gb.predict(x_test)

submit("gb_rmsle")