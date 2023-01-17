# Manipulation

import pandas as pd

import numpy as np



# Vizualisation

from IPython.display import display

import seaborn as sns

sns.set(style="white")

import matplotlib.pyplot as plt

%matplotlib inline



# Machine Learning

from sklearn import preprocessing
# Load data

train = pd.read_csv("../input/train.csv")



# First look at the data

print("Dimensions and first rows of the dataset :")

print(train.shape)

display(train.head())
# Check for duplicates

idsUnique = len(set(train.Id))

idsTotal = train.shape[0]

idsDupli = idsTotal - idsUnique

print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")
# Look at the target variable

print("We'll want to predict the last column : SalePrice")

print("The distribution is skewed on the right")

train.SalePrice.hist(bins = 50)
# Look at missing values

train.info()

print("--------------------")

print("Drop columns including too many missing values : Alley, FireplaceQu, PoolQC, Fence, MiscFeature")

print("We will infer values for columns with few NAs. New shape :")

train = train.drop(["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"], axis = 1)

print(train.shape)
# Handle missing values

print("For numerical columns, replace NAs with median")

print("For categorical columns, replace NAs with most frequent value")



# Numerical columns

train.LotFrontage.fillna(train.LotFrontage.median(), inplace = True)

train.MasVnrArea.fillna(train.MasVnrArea.median(), inplace = True)

train.GarageYrBlt.fillna(train.GarageYrBlt.median(), inplace = True)



# Categorical columns

train.MasVnrType.fillna(train.MasVnrType.value_counts().idxmax(), inplace = True)

train.BsmtQual.fillna(train.BsmtQual.value_counts().idxmax(), inplace = True)

train.BsmtCond.fillna(train.BsmtCond.value_counts().idxmax(), inplace = True)

train.BsmtExposure.fillna(train.BsmtExposure.value_counts().idxmax(), inplace = True)

train.BsmtFinType1.fillna(train.BsmtFinType1.value_counts().idxmax(), inplace = True)

train.BsmtFinType2.fillna(train.BsmtFinType2.value_counts().idxmax(), inplace = True)

train.Electrical.fillna(train.Electrical.value_counts().idxmax(), inplace = True)

train.GarageType.fillna(train.GarageType.value_counts().idxmax(), inplace = True)

train.GarageFinish.fillna(train.GarageFinish.value_counts().idxmax(), inplace = True)

train.GarageQual.fillna(train.GarageQual.value_counts().idxmax(), inplace = True)

train.GarageCond.fillna(train.GarageCond.value_counts().idxmax(), inplace = True)



print("Remaining NAs : " + str(train.isnull().values.sum()))
# For categorical variables, encode values as numbers so we can use those in correlation matrices

le = preprocessing.LabelEncoder()

le.fit(train.MSZoning)

train.MSZoning = le.transform(train.MSZoning)

le.fit(train.Street)

train.Street = le.transform(train.Street)

le.fit(train.LotShape)

train.LotShape = le.transform(train.LotShape)

le.fit(train.LandContour)

train.LandContour = le.transform(train.LandContour)

le.fit(train.Utilities)

train.Utilities = le.transform(train.Utilities)

le.fit(train.LotConfig)

train.LotConfig = le.transform(train.LotConfig)

le.fit(train.LandSlope)

train.LandSlope = le.transform(train.LandSlope)

le.fit(train.Neighborhood)

train.Neighborhood = le.transform(train.Neighborhood)

le.fit(train.Condition1)

train.Condition1 = le.transform(train.Condition1)

le.fit(train.Condition2)

train.Condition2 = le.transform(train.Condition2)

le.fit(train.BldgType)

train.BldgType = le.transform(train.BldgType)

le.fit(train.HouseStyle)

train.HouseStyle = le.transform(train.HouseStyle)

le.fit(train.RoofStyle)

train.RoofStyle = le.transform(train.RoofStyle)

le.fit(train.RoofMatl)

train.RoofMatl = le.transform(train.RoofMatl)

le.fit(train.Exterior1st)

train.Exterior1st = le.transform(train.Exterior1st)

le.fit(train.Exterior2nd)

train.Exterior2nd = le.transform(train.Exterior2nd)

le.fit(train.MasVnrType)

train.MasVnrType = le.transform(train.MasVnrType)

le.fit(train.ExterQual)

train.ExterQual = le.transform(train.ExterQual)

le.fit(train.ExterCond)

train.ExterCond = le.transform(train.ExterCond)

le.fit(train.BsmtQual)

train.BsmtQual = le.transform(train.BsmtQual)

le.fit(train.BsmtCond)

train.BsmtCond = le.transform(train.BsmtCond)

le.fit(train.BsmtExposure)

train.BsmtExposure = le.transform(train.BsmtExposure)

le.fit(train.BsmtFinType1)

train.BsmtFinType1 = le.transform(train.BsmtFinType1)

le.fit(train.BsmtFinType2)

train.BsmtFinType2 = le.transform(train.BsmtFinType2)

le.fit(train.Heating)

train.Heating = le.transform(train.Heating)

le.fit(train.HeatingQC)

train.HeatingQC = le.transform(train.HeatingQC)

le.fit(train.CentralAir)

train.CentralAir = le.transform(train.CentralAir)

le.fit(train.Electrical)

train.Electrical = le.transform(train.Electrical)

le.fit(train.KitchenQual)

train.KitchenQual = le.transform(train.KitchenQual)

le.fit(train.Functional)

train.Functional = le.transform(train.Functional)

le.fit(train.GarageType)

train.GarageType = le.transform(train.GarageType)

le.fit(train.GarageFinish)

train.GarageFinish = le.transform(train.GarageFinish)

le.fit(train.GarageQual)

train.GarageQual = le.transform(train.GarageQual)

le.fit(train.GarageCond)

train.GarageCond = le.transform(train.GarageCond)

le.fit(train.PavedDrive)

train.PavedDrive = le.transform(train.PavedDrive)

le.fit(train.SaleType)

train.SaleType = le.transform(train.SaleType)

le.fit(train.SaleCondition)

train.SaleCondition = le.transform(train.SaleCondition)



display(train.head())
# Plot a pretty correlation matrix

print("Diagonal correlation matrix for numerical variables (last row is target variable) :")

corr = train.corr()

mask = np.zeros_like(corr, dtype = np.bool)

mask[np.triu_indices_from(mask)] = True

plt.subplots(figsize = (14, 8))

sns.heatmap(corr, 

            mask = mask, 

            xticklabels = 3, 

            yticklabels = 3,

            linewidths = .5, 

            cbar_kws = {"shrink": .5})
# Find the variables most correlated to SalePrice

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

corr = corr.SalePrice

display(corr)
# Plot the 3 numerical variables most correlated with SalePrice

print("Most important variable for SalePrice is OverallQual : overall material and finish of the house")

sns.regplot(x = "OverallQual",

           y = "SalePrice", 

           data = train)

sns.plt.show()

print("2nd most important variable for SalePrice is GrLivArea : above grade (ground) living area square feet")

sns.regplot(x = "GrLivArea",

           y = "SalePrice", 

           data = train)

sns.plt.show()

print("3rd most important variable for SalePrice is GarageCars : size of garage in car capacity")

sns.regplot(x = "GarageCars",

           y = "SalePrice", 

           data = train)

sns.plt.show()