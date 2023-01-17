# data processing and visualization

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

# algorithm

from xgboost.sklearn import XGBRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.tree import DecisionTreeRegressor

# training训练

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
sns.distplot(train.SalePrice)
sns.distplot(np.log(train.SalePrice + 1))
all_data = pd.concat((train.drop(["SalePrice"], axis=1), test))

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100

all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)

plt.figure(figsize=(12, 6))

plt.xticks(rotation="90")

sns.barplot(x=all_data_na.index, y=all_data_na)
all_data[all_data.PoolArea != 0][["PoolArea", "PoolQC"]]
all_data[all_data.MiscVal > 10000][["MiscFeature", "MiscVal"]]
all_data[(all_data.GarageType.notnull()) & (all_data.GarageYrBlt.isnull())][["Neighborhood", "YearBuilt", "YearRemodAdd", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars", "GarageArea", "GarageQual", "GarageCond"]]
train.loc[[332, 948]][["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"]]
test.loc[[27, 580, 725, 757, 758, 888, 1064]][["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF", "BsmtFullBath", "BsmtHalfBath"]]
plt.scatter(train.Utilities, train.SalePrice)
y = train["SalePrice"]

y = np.log(y+1)
# PoolQC

test.loc[960, "PoolQC"] = "Fa"

test.loc[1043, "PoolQC"] = "Gd"

test.loc[1139, "PoolQC"] = "Fa"

 

# Garage

test.loc[666, "GarageYrBlt"] = 1979

test.loc[1116, "GarageYrBlt"] = 1979

 

test.loc[666, "GarageFinish"] = "Unf"

test.loc[1116, "GarageFinish"] = "Unf"

 

test.loc[1116, "GarageCars"] = 2

test.loc[1116, "GarageArea"] = 480

 

test.loc[666, "GarageQual"] = "TA"

test.loc[1116, "GarageQual"] = "TA"

 

test.loc[666, "GarageCond"] = "TA"

test.loc[1116, "GarageCond"] = "TA"
# PoolQC

train = train.fillna({"PoolQC": "None"})

test = test.fillna({"PoolQC": "None"})

 

# Alley

train = train.fillna({"Alley": "None"})

test = test.fillna({"Alley": "None"})

 

# FireplaceQu

train = train.fillna({"FireplaceQu": "None"})

test = test.fillna({"FireplaceQu": "None"})

 

# LotFrontage

train = train.fillna({"LotFrontage": 0})

test = test.fillna({"LotFrontage": 0})

 

# Garage

train = train.fillna({"GarageType": "None"})

test = test.fillna({"GarageType": "None"})

train = train.fillna({"GarageYrBlt": 0})

test = test.fillna({"GarageYrBlt": 0})

train = train.fillna({"GarageFinish": "None"})

test = test.fillna({"GarageFinish": "None"})

test = test.fillna({"GarageCars": 0})

test = test.fillna({"GarageArea": 0})

train = train.fillna({"GarageQual": "None"})

test = test.fillna({"GarageQual": "None"})

train = train.fillna({"GarageCond": "None"})

test = test.fillna({"GarageCond": "None"})

 

# Bsmt

train = train.fillna({"BsmtQual": "None"})

test = test.fillna({"BsmtQual": "None"})

train = train.fillna({"BsmtCond": "None"})

test = test.fillna({"BsmtCond": "None"})

train = train.fillna({"BsmtExposure": "None"})

test = test.fillna({"BsmtExposure": "None"})

train = train.fillna({"BsmtFinType1": "None"})

test = test.fillna({"BsmtFinType1": "None"})

train = train.fillna({"BsmtFinSF1": 0})

test = test.fillna({"BsmtFinSF1": 0})

train = train.fillna({"BsmtFinType2": "None"})

test = test.fillna({"BsmtFinType2": "None"})

test = test.fillna({"BsmtFinSF2": 0})

test = test.fillna({"BsmtUnfSF": 0})

test = test.fillna({"TotalBsmtSF": 0})

test = test.fillna({"BsmtFullBath": 0})

test = test.fillna({"BsmtHalfBath": 0})

 

# MasVnr

train = train.fillna({"MasVnrType": "None"})

test = test.fillna({"MasVnrType": "None"})

train = train.fillna({"MasVnrArea": 0})

test = test.fillna({"MasVnrArea": 0})

 

# MiscFeature,Fence,Utilities

train = train.drop(["Fence", "MiscFeature", "Utilities"], axis=1)

test = test.drop(["Fence", "MiscFeature", "Utilities"], axis=1)

 

# other

test = test.fillna({"MSZoning": "RL"})

test = test.fillna({"Exterior1st": "VinylSd"})

test = test.fillna({"Exterior2nd": "VinylSd"})

train = train.fillna({"Electrical": "SBrkr"})

test = test.fillna({"KitchenQual": "TA"})

test = test.fillna({"Functional": "Typ"})

test = test.fillna({"SaleType": "WD"})
train_dummies = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[: train.shape[0]]

test_dummies = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[train.shape[0]:]
rr = Ridge(alpha=10)

rr.fit(train_dummies, y)

np.sqrt(-cross_val_score(rr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
y_pred = rr.predict(train_dummies)

resid = y - y_pred

mean_resid = resid.mean()

std_resid = resid.std()

z = (resid - mean_resid) / std_resid

z = np.array(z)

outliers1 = np.where(abs(z) > abs(z).std() * 3)[0]

outliers1
plt.figure(figsize=(6, 6))

plt.scatter(y, y_pred)

plt.scatter(y.iloc[outliers1], y_pred[outliers1])

plt.plot(range(10, 15), range(10, 15), color="red")
er = ElasticNet(alpha=0.001, l1_ratio=0.58)

er.fit(train_dummies, y)

np.sqrt(-cross_val_score(rr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
y_pred = er.predict(train_dummies)

resid = y - y_pred

mean_resid = resid.mean()

std_resid = resid.std()

z = (resid - mean_resid) / std_resid

z = np.array(z)

outliers2 = np.where(abs(z) > abs(z).std() * 3)[0]

outliers2
plt.figure(figsize=(6, 6))

plt.scatter(y, y_pred)

plt.scatter(y.iloc[outliers2], y_pred[outliers2])

plt.plot(range(10, 15), range(10, 15), color="red")
outliers = []

for i in outliers1:

    for j in outliers2:

        if i == j:

            outliers.append(i)

outliers
train = train.drop([30, 88, 142, 277, 328, 410, 462, 495, 523, 533, 581, 588, 628, 632, 681, 688, 710, 714, 728, 774, 812, 874, 898, 916, 968, 970, 1181, 1182, 1298, 1324, 1383, 1423, 1432, 1453])

y = train["SalePrice"]

y = np.log(y+1)
train_dummies = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[: train.shape[0]]

test_dummies = pd.get_dummies(pd.concat((train.drop(["SalePrice", "Id"], axis=1), test.drop(["Id"], axis=1)), axis=0)).iloc[train.shape[0]:]
gbr = GradientBoostingRegressor(max_depth=4, n_estimators=150)

gbr.fit(train_dummies, y)

np.sqrt(-cross_val_score(gbr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
xgbr = XGBRegressor(max_depth=5, n_estimators=400)

xgbr.fit(train_dummies, y)

np.sqrt(-cross_val_score(xgbr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
lsr = Lasso(alpha=0.00047)

lsr.fit(train_dummies, y)

np.sqrt(-cross_val_score(lsr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
rr = Ridge(alpha=13)

rr.fit(train_dummies, y)

np.sqrt(-cross_val_score(rr, train_dummies, y, cv=5, scoring="neg_mean_squared_error")).mean()
train_predict = 0.1 * gbr.predict(train_dummies) + 0.3 * xgbr.predict(train_dummies) + 0.3 * lsr.predict(train_dummies) + 0.3 * rr.predict(train_dummies)
plt.figure(figsize=(6, 6))

plt.scatter(y, train_predict)

plt.plot(range(10, 15), range(10, 15), color="red")
q1 = pd.DataFrame(train_predict).quantile(0.0042)

pre_df = pd.DataFrame(train_predict)

pre_df["SalePrice"] = train_predict

pre_df = pre_df[["SalePrice"]]

pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] = pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] *0.99

train_predict = np.array(pre_df.SalePrice)

plt.figure(figsize=(6, 6))

plt.scatter(y, train_predict)

plt.plot(range(10, 15), range(10, 15), color="red")
test_predict = 0.1 * gbr.predict(test_dummies) + 0.3 * xgbr.predict(test_dummies) + 0.3 * lsr.predict(test_dummies) + 0.3 * rr.predict(test_dummies)

q1 = pd.DataFrame(test_predict).quantile(0.0042)

pre_df = pd.DataFrame(test_predict)

pre_df["SalePrice"] = test_predict

pre_df = pre_df[["SalePrice"]]

pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] = pre_df.loc[pre_df.SalePrice <= q1[0], "SalePrice"] *0.96

test_predict = np.array(pre_df.SalePrice)

sample_submission["SalePrice"] = np.exp(test_predict)-1

sample_submission.to_csv("1.csv", index=False)