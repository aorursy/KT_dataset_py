# Author: Alaa Awad

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import tensorflow as tf



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr



%config InlineBackend.figure_format = 'png' #set 'png' here when working on notebook

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
print("train : " + str(train.shape))

print("test : " + str(test.shape))
# Check for duplicates

idsUnique = len(set(train.Id))

idsTotal = train.shape[0]

idsDupli = idsTotal - idsUnique

print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")



# Drop Id column

train.drop("Id", axis = 1, inplace = True)
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()



train = train[train.GrLivArea < 4000]
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

prices = pd.DataFrame({"price":train["SalePrice"], "log(price + 1)":np.log1p(train["SalePrice"])})

prices.hist()
sns.distplot(train["SalePrice"], bins=30, kde = False, color = 'b', hist_kws={'alpha': 0.9})
# Handle missing values for features where median/mean or most common value doesn't make sense



# Alley : data description says NA means "no alley access"

train.loc[:, "Alley"] = train.loc[:, "Alley"].fillna("None")

# BedroomAbvGr : NA most likely means 0

train.loc[:, "BedroomAbvGr"] = train.loc[:, "BedroomAbvGr"].fillna(0)

# BsmtQual etc : data description says NA for basement features is "no basement"

train.loc[:, "BsmtQual"] = train.loc[:, "BsmtQual"].fillna("No")

train.loc[:, "BsmtCond"] = train.loc[:, "BsmtCond"].fillna("No")

train.loc[:, "BsmtExposure"] = train.loc[:, "BsmtExposure"].fillna("No")

train.loc[:, "BsmtFinType1"] = train.loc[:, "BsmtFinType1"].fillna("No")

train.loc[:, "BsmtFinType2"] = train.loc[:, "BsmtFinType2"].fillna("No")

train.loc[:, "BsmtFullBath"] = train.loc[:, "BsmtFullBath"].fillna(0)

train.loc[:, "BsmtHalfBath"] = train.loc[:, "BsmtHalfBath"].fillna(0)

train.loc[:, "BsmtUnfSF"] = train.loc[:, "BsmtUnfSF"].fillna(0)

# CentralAir : NA most likely means No

train.loc[:, "CentralAir"] = train.loc[:, "CentralAir"].fillna("N")

# Condition : NA most likely means Normal

train.loc[:, "Condition1"] = train.loc[:, "Condition1"].fillna("Norm")

train.loc[:, "Condition2"] = train.loc[:, "Condition2"].fillna("Norm")

# EnclosedPorch : NA most likely means no enclosed porch

train.loc[:, "EnclosedPorch"] = train.loc[:, "EnclosedPorch"].fillna(0)

# External stuff : NA most likely means average

train.loc[:, "ExterCond"] = train.loc[:, "ExterCond"].fillna("TA")

train.loc[:, "ExterQual"] = train.loc[:, "ExterQual"].fillna("TA")

# Fence : data description says NA means "no fence"

train.loc[:, "Fence"] = train.loc[:, "Fence"].fillna("No")

# FireplaceQu : data description says NA means "no fireplace"

train.loc[:, "FireplaceQu"] = train.loc[:, "FireplaceQu"].fillna("No")

train.loc[:, "Fireplaces"] = train.loc[:, "Fireplaces"].fillna(0)

# Functional : data description says NA means typical

train.loc[:, "Functional"] = train.loc[:, "Functional"].fillna("Typ")

# GarageType etc : data description says NA for garage features is "no garage"

train.loc[:, "GarageType"] = train.loc[:, "GarageType"].fillna("No")

train.loc[:, "GarageFinish"] = train.loc[:, "GarageFinish"].fillna("No")

train.loc[:, "GarageQual"] = train.loc[:, "GarageQual"].fillna("No")

train.loc[:, "GarageCond"] = train.loc[:, "GarageCond"].fillna("No")

train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)

train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)

# HalfBath : NA most likely means no half baths above grade

train.loc[:, "HalfBath"] = train.loc[:, "HalfBath"].fillna(0)

# HeatingQC : NA most likely means typical

train.loc[:, "HeatingQC"] = train.loc[:, "HeatingQC"].fillna("TA")

# KitchenAbvGr : NA most likely means 0

train.loc[:, "KitchenAbvGr"] = train.loc[:, "KitchenAbvGr"].fillna(0)

# KitchenQual : NA most likely means typical

train.loc[:, "KitchenQual"] = train.loc[:, "KitchenQual"].fillna("TA")

# LotFrontage : NA most likely means no lot frontage

train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)

# LotShape : NA most likely means regular

train.loc[:, "LotShape"] = train.loc[:, "LotShape"].fillna("Reg")

# MasVnrType : NA most likely means no veneer

train.loc[:, "MasVnrType"] = train.loc[:, "MasVnrType"].fillna("None")

train.loc[:, "MasVnrArea"] = train.loc[:, "MasVnrArea"].fillna(0)

# MiscFeature : data description says NA means "no misc feature"

train.loc[:, "MiscFeature"] = train.loc[:, "MiscFeature"].fillna("No")

train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)

# OpenPorchSF : NA most likely means no open porch

train.loc[:, "OpenPorchSF"] = train.loc[:, "OpenPorchSF"].fillna(0)

# PavedDrive : NA most likely means not paved

train.loc[:, "PavedDrive"] = train.loc[:, "PavedDrive"].fillna("N")

# PoolQC : data description says NA means "no pool"

train.loc[:, "PoolQC"] = train.loc[:, "PoolQC"].fillna("No")

train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)

# SaleCondition : NA most likely means normal sale

train.loc[:, "SaleCondition"] = train.loc[:, "SaleCondition"].fillna("Normal")

# ScreenPorch : NA most likely means no screen porch

train.loc[:, "ScreenPorch"] = train.loc[:, "ScreenPorch"].fillna(0)

# TotRmsAbvGrd : NA most likely means 0

train.loc[:, "TotRmsAbvGrd"] = train.loc[:, "TotRmsAbvGrd"].fillna(0)

# Utilities : NA most likely means all public utilities

train.loc[:, "Utilities"] = train.loc[:, "Utilities"].fillna("AllPub")

# WoodDeckSF : NA most likely means no wood deck

train.loc[:, "WoodDeckSF"] = train.loc[:, "WoodDeckSF"].fillna(0)
corr = train.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(9, 9))

sns.heatmap(corr, vmax=1, square=True)
cor_dict = corr['SalePrice'].to_dict()

del cor_dict['SalePrice']

print("List the numerical features decendingly by their correlation with Sale Price:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))