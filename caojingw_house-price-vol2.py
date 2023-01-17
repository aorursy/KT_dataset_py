import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

full = pd.concat((train,test))
print (train.shape)

print (test.shape)
full.head()
print (train.shape)

print (test.shape)

print (full.shape)
missing = pd.DataFrame({"miss_percentage": full.isna().sum() / full.shape[0]})

missing[missing.miss_percentage>0].sort_values(by = 'miss_percentage', ascending=False)
# PoolQC: NA means no pool

full["PoolQC"] = full["PoolQC"].fillna("None")

#MiscFeature

full["MiscFeature"] = full["MiscFeature"].fillna("None")

#Alley

full["Alley"] = full["Alley"].fillna("None")

#Fence

full["Fence"] = full["Fence"].fillna("None")

#FireplaceQu

full["FireplaceQu"] = full["FireplaceQu"].fillna("None")

#LotFrontage

full["LotFrontage"] = full["LotFrontage"].fillna(full["LotFrontage"].median())

#GarageType, GarageFinish, GarageQual and GarageCond

for i in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:

    full[i] = full[i].fillna("None")

#GarageYrBlt, GarageArea and GarageCars, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtFullBath and BsmtHalfBath, MasVnrArea 

for i in ["GarageYrBlt", "GarageArea", "GarageCars", "GarageCond", 

          "BsmtFinSF1","BsmtFinSF2","BsmtUnfSF", "TotalBsmtSF","BsmtFullBath", "BsmtHalfBath","MasVnrArea"]:

    full[i] = full[i].fillna(0)

#BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2,MasVnrType -> None

for i in ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1","BsmtFinType2","MasVnrType"]:

    full[i] = full[i].fillna("None")

#MSZoning 

full["MSZoning"] = full["MSZoning"].fillna(full["MSZoning"].mode()[0])

#Utilities -> remove

full = full.drop(['Utilities'], axis=1)

#Functional 

full["Functional"] = full["Functional"].fillna("Typ")

#Electrical 

full['Electrical'] = full['Electrical'].fillna(full['Electrical'].mode()[0])

#KitchenQual

full['KitchenQual'] = full['KitchenQual'].fillna(full['KitchenQual'].mode()[0])

#Exterior1st and Exterior2nd

full['Exterior1st'] = full['Exterior1st'].fillna(full['Exterior1st'].mode()[0])

full['Exterior2nd'] = full['Exterior2nd'].fillna(full['Exterior2nd'].mode()[0])

#SaleType

full['SaleType'] = full['SaleType'].fillna(full['SaleType'].mode()[0])

#MSSubClass

full['MSSubClass'] = full['MSSubClass'].fillna("None")
# Final check

full.isna().sum()[full.isna().sum() > 0]
# Split full into train and test and there will be no missing values

train_df = full[:train.shape[0]]

test_df = full[train.shape[0]:]

train_df.drop(['Id'], axis=1, inplace=True)

test_df.drop(['Id','SalePrice'], axis=1, inplace=True)

print (train_df.shape)

print (test_df.shape)
import seaborn as sns

from matplotlib import pyplot as plt
sns.distplot(train_df['SalePrice']);
#We use the numpy fuction log1p which  applies log(1+x)

train_df["SalePrice"] = np.log1p(train_df["SalePrice"])



#Check the new distribution 

sns.distplot(train['SalePrice']);
plt.figure(figsize=(20,20))

sns.heatmap(train_df.corr(),annot=True,cmap="RdYlGn",fmt=".1f")
cor_df = train_df.corr()

df=pd.DataFrame({'col': cor_df.index,

    'corr': np.abs(cor_df['SalePrice'])}).sort_values(by='corr', ascending=False)



keep_col = df.col[df['corr']>=0.1]



train = train_df[keep_col]





test_df['SalePrice'] = 0

test_X = test_df[train.columns]

test_df.drop(['SalePrice'], axis = 1, inplace = True)
train_df = pd.get_dummies(train_df)

test_df = pd.get_dummies(test_df)
col_keep = [col for col in train_df.columns

             if col in test_df.columns]
train_y = train_df.SalePrice

train_X = train_df[col_keep]
print (train_df.shape)
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
rf = RandomForestRegressor()

rf.fit(train_X,train_y)

xgb = XGBRegressor()

xgb.fit(train_X,train_y)
rf_score = -1 * cross_val_score(rf, train_X, train_y, scoring="neg_mean_squared_error", cv = 5)

xgb_score = -1 * cross_val_score(xgb, train_X, train_y, scoring="neg_mean_squared_error", cv = 5)
print (rf_score.mean())

print (xgb_score.mean())
output = pd.DataFrame({'Id':  test.Id,

                      'SalePrice': np.expm1(xgb.predict(test_df))})



output.to_csv("submission.csv", index=False)