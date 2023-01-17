# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as pl

from matplotlib import cm as cm

import plotly.plotly as py



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/train.csv")


train.describe()
train.count()
train.shape
train.columns
train.dtypes
train.corr()
fig = pl.figure()

ax1 = fig.add_subplot(111)

cmap = cm.get_cmap('jet', 80)

cax = ax1.imshow(train.corr(), interpolation="nearest", cmap=cmap)

ax1.grid(True)

pl.title('Abalone Feature Correlation')

#labels=['Id',	'MSSubClass',	'LotFrontage',	'LotArea',	'OverallQual',	'OverallCond',	'YearBuilt',	'YearRemodAdd',	'MasVnrArea',	'BsmtFinSF1',	'BsmtFinSF2',	'BsmtUnfSF',	'TotalBsmtSF',	'1stFlrSF',	'2ndFlrSF',	'LowQualFinSF',	'GrLivArea',	'BsmtFullBath',	'BsmtHalfBath',	'FullBath',	'HalfBath',	'BedroomAbvGr',	'KitchenAbvGr',	'TotRmsAbvGrd',	'Fireplaces',	'GarageYrBlt',	'GarageCars',	'GarageArea',	'WoodDeckSF',	'OpenPorchSF',	'EnclosedPorch',	'3SsnPorch',	'ScreenPorch',	'PoolArea',	'MiscVal',	'MoSold',	'YrSold',	'SalePrice',]

#ax1.set_xticklabels(labels,fontsize=5)

#ax1.set_yticklabels(labels,fontsize=5)

# Add colorbar, make sure to specify tick locations to match desired ticklabels

fig.colorbar(cax, ticks=[.75,.8,.85,.90,.95,1])

pl.show()


pl.plot(train["YearBuilt"], train["GarageYrBlt"], "o")

traincor=train[["YearBuilt","GarageYrBlt"]]

traincor.corr()
pl.plot(train["TotalBsmtSF"], train["1stFlrSF"], "o")

traincor=train[["TotalBsmtSF","1stFlrSF"]]

traincor.corr()


pl.plot(train["GarageCars"], train["GarageArea"], "o")

traincor=train[["GarageCars","GarageArea"]]

traincor.corr()
pl.plot(train["SalePrice"], train["MiscVal"], "o")

traincor=train[["SalePrice","MiscVal"]]

traincor.corr()
Alley2=train['Alley'].fillna('NoAlley')

del train["Alley"]

train["Alley"]=Alley2



MiscFeature2=train['MiscFeature'].fillna('NoMiscFeature')

del train["MiscFeature"]

train["MiscFeature"]=MiscFeature2





Fence2=train['Fence'].fillna('NoFence')

del train["Fence"]

train["Fence"]=Fence2





PoolQC2=train['PoolQC'].fillna('NoPool')

del train["PoolQC"]

train["PoolQC"]=PoolQC2



GarageCond2=train['GarageCond'].fillna('NoGarage')

del train["GarageCond"]

train["GarageCond"]=GarageCond2



GarageQual2=train['GarageQual'].fillna('NoGarage')

del train["GarageQual"]

train["GarageQual"]=GarageCond2



GarageFinish2=train['GarageFinish'].fillna('NoGarage')

del train["GarageFinish"]

train["GarageFinish"]=GarageFinish2



GarageType2=train['GarageType'].fillna('NoGarage')

del train["GarageType"]

train["GarageType"]=GarageType2



FireplaceQu2=train['FireplaceQu'].fillna('NoFireplace')

del train["FireplaceQu"]

train["FireplaceQu"]=FireplaceQu2



BsmtFinType22=train['BsmtFinType2'].fillna('NoBasement')

del train["BsmtFinType2"]

train["BsmtFinType2"]=BsmtFinType22



BsmtFinType12=train['BsmtFinType1'].fillna('NoBasement')

del train["BsmtFinType1"]

train["BsmtFinType1"]=BsmtFinType12



BsmtCond2=train['BsmtCond'].fillna('NoBasement')

del train["BsmtCond"]

train["BsmtCond"]=BsmtCond2

  

BsmtQual2=train['BsmtQual'].fillna('NoBasement')

del train["BsmtQual"]

train["BsmtQual"]=BsmtQual2















train.count()
train["LotFrontage"].groupby(train["MSZoning"]).describe()
pl.plot(train["LotFrontage"], train["LotArea"], "o")

traincor=train[["LotFrontage","LotArea"]]

traincor.corr()
pl.plot(train["LotArea"], train["LotFrontage"], "o")

traincor=train[["LotArea","LotFrontage"]]

traincor.corr()
pl.plot(train["LotFrontage"], train["SalePrice"], "o")

traincor=train[["LotFrontage","SalePrice"]]

traincor.corr()
pl.plot(train["SalePrice"], train["LotFrontage"], "o")

traincor=train[["SalePrice","LotFrontage"]]

traincor.corr()
LFm=train["LotFrontage"].mean()

LotFrontage2=train['LotFrontage'].fillna(LFm)

del train["LotFrontage"]

train["LotFrontage"]=LotFrontage2
subset=train[train.GarageYrBlt!=train.YearBuilt ] 

subset=subset[subset.GarageYrBlt.isnull()==False]

subset2=train[train.GarageYrBlt==train.YearBuilt] 
subset[["GarageYrBlt","YearBuilt"]].count()
pl.plot(subset["GarageYrBlt"], subset["SalePrice"], "o")

traincor=subset[["GarageYrBlt","SalePrice"]]

traincor.corr()
pl.plot(subset2["GarageYrBlt"], subset2["SalePrice"], "o")

traincor=subset2[["GarageYrBlt","SalePrice"]]

traincor.corr()
pl.plot(train["GarageYrBlt"], train["SalePrice"], "o")

traincor=train[["GarageYrBlt","SalePrice"]]

traincor.corr()
GarageYrBlt2=train['GarageYrBlt'].fillna(0)



train["GarageYrBlt2"]=GarageYrBlt2









GarageYrBlt2=train['GarageYrBlt'].fillna(train.YearBuilt)



train["GarageYrBlt3"]=GarageYrBlt2
pl.plot(train["GarageYrBlt2"], train["SalePrice"], "o")

traincor=train[["GarageYrBlt2","SalePrice"]]

traincor.corr()
pl.plot(train["GarageYrBlt3"], train["SalePrice"], "o")

traincor=train[["GarageYrBlt3","SalePrice"]]

traincor.corr()
del train["GarageYrBlt2"]

train.count()


MasVnrType2=train['MasVnrType'].fillna("none")

del train["MasVnrType"]

train["MasVnrType"]=MasVnrType2



MasVnrArea2=train['MasVnrArea'].fillna(0.0)

del train["MasVnrArea"]

train["MasVnrArea"]=MasVnrArea2



train["BsmtExposure"]=train['BsmtExposure'].fillna("NoBasement")

train[["BsmtExposure","BsmtCond"]][train.BsmtExposure=="NoBasement"]
train=train.drop(train.index[948])
pd.options.display.max_rows = 999

pd.options.display.max_columns=999
train=train.dropna()