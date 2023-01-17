import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib as mpl

from scipy.stats import skew

import matplotlib.pyplot as plt

from IPython.display import display, HTML

%matplotlib inline
train=pd.read_csv('../input/train.csv')

print(len(train))

train.head(2)
numeric_feats = train.dtypes[train.dtypes != "object"].index

numeric_feats
# define plot function, and in this function, we will calculate the skew of X and take the log1p of y

def plot_outlier(x,y):

    tmp=x.dropna()

    skew_value=skew(tmp)

    y=np.log1p(y)

    print('sample lengh: %s   and skew: %s'%(len(x),skew_value))

    fig,axs=plt.subplots(1,2,figsize=(8,3))

    sns.boxplot(x,orient='v',ax=axs[0])

    sns.regplot(x,y,ax=axs[1])

    plt.tight_layout()
# LotFrontage

plot_outlier(train.LotFrontage,train.SalePrice)
# we can see the skew value is a little bit high 

# it seems that there are two outlier about LotFrontage, let's remove them and replot 

train=train[train.LotFrontage<300]

plot_outlier(train.LotFrontage,train.SalePrice)

# from the skew value and the plot, it seems better than before, the cost is we lose some sample :(
# LotArea

plot_outlier(train.LotArea,train.SalePrice)
# the same thing we do for LotArea, I select threshold from 7000->5000->3000

train=train[train.LotArea<30000]

plot_outlier(train.LotArea,train.SalePrice)
# OverallQual, from the plot, we can tell there is a strong linear relationship between them

plot_outlier(train.OverallQual,train.SalePrice)
# OverallCond, normally rating features do not have outliers, but let's just look at the plot

plot_outlier(train.OverallCond,train.SalePrice)

# it's strange for trend, it seems that the higger rating the cheaper it is 
# YearBuilt, it seems that the newer of the house the higher of the price

plot_outlier(2016-train.YearBuilt,train.SalePrice)
# YearRemodAdd, same as YearBuilt

plot_outlier(2016-train.YearRemodAdd,train.SalePrice)
# MasVnrArea, 

plot_outlier(train.MasVnrArea,train.SalePrice)
# there are a lot of 0 in sample

train=train[train.MasVnrArea<1500]

plot_outlier(train.MasVnrArea,train.SalePrice)
# BsmtFinSF1, 

plot_outlier(train.BsmtFinSF1,train.SalePrice)
train=train[train.BsmtFinSF1<2000]

plot_outlier(train.BsmtFinSF1,train.SalePrice)
# BsmtFinSF2, there are a lot of zero in sample, and the regression line likes a horizontal line, may be this feature is not so important

plot_outlier(train.BsmtFinSF2,train.SalePrice)
# let's add BsmtFinSF1 and BsmtFinSF2 together

totalBsmtFinSF=train.BsmtFinSF1+train.BsmtFinSF2

plot_outlier(totalBsmtFinSF,train.SalePrice)

# nothing special
# BsmtUnfSF

plot_outlier(train.BsmtUnfSF,train.SalePrice)

# for me, this feature also strange:

# 1. why the more Unfinished area, the higher of the price?
# TotalBsmtSF

plot_outlier(train.TotalBsmtSF,train.SalePrice)

# question:

# 1. does TotalBsmtSF include the BsmtUnfSF? if so let's plot the finished feet
train=train[train.TotalBsmtSF<3000]

plot_outlier(train.TotalBsmtSF,train.SalePrice)
# finished square feet

BsmtFSF=train.TotalBsmtSF-train.BsmtUnfSF

plot_outlier(BsmtFSF,train.SalePrice)
# 1stFlrSF

plot_outlier(train.loc[:,'1stFlrSF'],train.SalePrice)
# 2ndFlrSF

plot_outlier(train.loc[:,'2ndFlrSF'],train.SalePrice)

# we can see from the plot, if there is no zero, the line will be more slope
# let's add the 1stFlrSF and 2stFlrSF together

totalFlrSF=train.loc[:,'1stFlrSF']+train.loc[:,'2ndFlrSF']

plot_outlier(totalFlrSF,train.SalePrice)
# LowQualFinSF

plot_outlier(train.LowQualFinSF,train.SalePrice)

# maybe not important feature
# GrLivArea

plot_outlier(train.GrLivArea,train.SalePrice)
train=train[train.GrLivArea<4000]

plot_outlier(train.GrLivArea,train.SalePrice)
plot_outlier(train.BsmtFullBath,train.SalePrice)
plot_outlier(train.BsmtHalfBath,train.SalePrice)
plot_outlier(train.FullBath,train.SalePrice)
plot_outlier(train.HalfBath,train.SalePrice)
# all the bathroom

totalBath=train.BsmtFullBath+train.BsmtHalfBath+train.FullBath+train.HalfBath

plot_outlier(totalBath,train.SalePrice)
plot_outlier(train.BedroomAbvGr,train.SalePrice)
plot_outlier(train.TotRmsAbvGrd,train.SalePrice)
plot_outlier(train.Fireplaces,train.SalePrice)
plot_outlier(2016-train.GarageYrBlt,train.SalePrice)
plot_outlier(train.GarageCars,train.SalePrice)
plot_outlier(train.GarageArea,train.SalePrice)
train=train[train.GarageArea<1230]

plot_outlier(train.GarageArea,train.SalePrice)
plot_outlier(train.WoodDeckSF,train.SalePrice)
train=train[train.WoodDeckSF<600]

plot_outlier(train.WoodDeckSF,train.SalePrice)
plot_outlier(train.OpenPorchSF,train.SalePrice)
train=train[train.OpenPorchSF<500]

plot_outlier(train.OpenPorchSF,train.SalePrice)
plot_outlier(train.EnclosedPorch,train.SalePrice)
train=train[train.EnclosedPorch<350]

plot_outlier(train.EnclosedPorch,train.SalePrice)
plot_outlier(train.loc[:,'3SsnPorch'],train.SalePrice)
plot_outlier(train.ScreenPorch,train.SalePrice)
totalPorch=train.OpenPorchSF+train.EnclosedPorch+train.loc[:,'3SsnPorch']+train.ScreenPorch

plot_outlier(totalPorch,train.SalePrice)
plot_outlier(train.MoSold ,train.SalePrice)
plot_outlier(train.YrSold ,train.SalePrice)