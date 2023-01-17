import numpy as np 
import pandas as pd 
import seaborn as sns
from matplotlib import pyplot
import os
print(os.listdir("../input"))
house_price = pd.read_csv("../input/train.csv")
house_price.head()
house_price_test = pd.read_csv("../input/test.csv")
house_price_test.head()
house_price.columns
house_price.describe()
factors = ["MSSubClass","LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", 
           "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", 
           "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr",
           "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces", "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF",
           "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"]
for factor in factors:
    house_price.plot.scatter(x=factor, y="SalePrice", title=factor, figsize=(12, 6), fontsize=16)
    sns.despine(bottom=True, left=True)
fig, ax = pyplot.subplots(2, 2, figsize=(20,20))
house_price.plot.scatter(x="YearBuilt", y="SalePrice", ax=ax[0][0])
sns.violinplot(
    x='YearBuilt',
    y='SalePrice',
    data=house_price,
    ax=ax[0][1]
)
house_price.plot.scatter(x="YearRemodAdd", y="SalePrice", ax=ax[1][0])
sns.violinplot(
    x='YearRemodAdd',
    y='SalePrice',
    data=house_price,
    ax=ax[1][1]
)

fig, ax = pyplot.subplots(2, 1)
house_price.plot.scatter(x="MasVnrArea", y="SalePrice", ax=ax[0])
house_price.plot.scatter(x="GrLivArea", y="SalePrice", ax=ax[1])
fig, ax = pyplot.subplots(2, 1)
house_price.plot.scatter(x="LotFrontage", y="SalePrice", ax=ax[0])
house_price.plot.scatter(x="LotArea", y="SalePrice", ax=ax[1])
fig, ax = pyplot.subplots(2, 1)
house_price.plot.scatter(x="GarageArea", y="SalePrice", ax=ax[0])
sns.violinplot(
    x='GarageCars',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
sns.violinplot(
    x='OverallQual',
    y='SalePrice',
    data=house_price,
    ax=ax[0]
)
sns.violinplot(
    x='OverallCond',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["MSSubClass"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='MSSubClass',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["SaleType"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='SaleType',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["GarageType"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='GarageType',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["Fireplaces"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Fireplaces',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["Functional"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Functional',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["KitchenQual"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='KitchenQual',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["Electrical"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Electrical',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["CentralAir"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='CentralAir',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["Heating"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Heating',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["HouseStyle"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='HouseStyle',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["BldgType"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='BldgType',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["Condition2"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Condition2',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1, figsize=(20,10))
house_price["Condition1"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Condition1',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1, figsize=(20,10))
house_price["Neighborhood"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Neighborhood',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["LandSlope"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='LandSlope',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["Street"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Street',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["Utilities"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='Utilities',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["LandContour"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='LandContour',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["LotShape"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='LotShape',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
fig, ax = pyplot.subplots(2, 1)
house_price["MSZoning"].value_counts().plot.bar(ax=ax[0])
sns.violinplot(
    x='MSZoning',
    y='SalePrice',
    data=house_price,
    ax=ax[1]
)
predictors = ["GrLivArea", "LotFrontage", "LotArea", "GarageArea", "GarageCars", "GarageType", "OverallQual",
              "Fireplaces", "KitchenQual", "Electrical", "CentralAir", "Heating", "HouseStyle", "Condition1",
              "Condition2", "Neighborhood", "LotShape", "MSZoning"
             ]
X = house_price[predictors]
y = house_price.SalePrice
from sklearn.tree import DecisionTreeRegressor

melbourne_model = DecisionTreeRegressor()
melbourne_model.fit(X, y)

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))