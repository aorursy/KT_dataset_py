from pandas import DataFrame, Series

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import statsmodels.api as sm

from sklearn.cross_validation import train_test_split



train = pd.read_csv("../input/train.csv")
# FEATURE ENGINEERING?

train["HasAlley"] = 1 - pd.isnull(train["Alley"])

train["HasFireplace"] = 0 + train["Fireplaces"] > 0

train["HasBsmt"] = 1 - pd.isnull(train["BsmtQual"])

train["HasFence"] = 1 - pd.isnull(train["Fence"])

train["HasPool"] = 1 - pd.isnull(train["PoolQC"])



def quality(s):

    qual = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, "Po": 1, 0: 0, "No": 3, "Mn": 3, "Av": 3}

    s = s.fillna(0)

    return [qual[e] for e in s]



train["ExterQual"] = quality(train["ExterQual"])

train["ExterCond"] = quality(train["ExterCond"])

train["BsmtQual"] = quality(train["BsmtQual"])

train["BsmtCond"] = quality(train["BsmtCond"])

train["BsmtExposure"] = quality(train["BsmtExposure"])

train["HeatingQC"] = quality(train["HeatingQC"])

train["KitchenQual"] = quality(train["KitchenQual"])

train["FireplaceQu"] = quality(train["FireplaceQu"])

train["GarageQual"] = quality(train["GarageQual"])

train["GarageCond"] = quality(train["GarageCond"])

train["PoolQC"] = quality(train["PoolQC"])
# fills na values with medians grouped by neighborhood

train = train.fillna(train.groupby('Neighborhood').median())

# select only numerical data

train = train.select_dtypes(include = ['float64', 'int64'])

train.head()
# serena's thing

correlations = train.corr()

for x in correlations.columns:

    for y in correlations.columns:

        if x < y and correlations[x][y] > .8 or correlations[x][y] < -.8:

            print(x, y, correlations[x][y])



print("low correlations!")

for x in correlations.columns:

    if correlations["SalePrice"][x] < .1 and correlations["SalePrice"][x] > -.1:

        print(x, correlations["SalePrice"][x])
def rightformat(predictions):

    n = 1461

    print("Id,SalePrice")

    for p in predictions:

        print(str(n) + "," + str(p))

        n += 1
