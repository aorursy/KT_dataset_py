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
pd.options.display.max_rows = 999

pd.options.display.max_columns=999

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
pl.plot(train["TotalBsmtSF"], train["1stFlrSF"], "o")


pl.plot(train["GarageCars"], train["GarageArea"], "o")
pl.plot(train["SalePrice"], train["MiscVal"], "o")

traincor=train[["SalePrice","MiscVal"]]

traincor.corr()
Alley2=train['Alley'].fillna('NoAlley')

train["Alley2"]=Alley2
train.count()