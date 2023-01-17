# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Other libraries import...

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")



print("The shape train data is " + str(train.shape))

# See how the data is like...

print(train.describe())
# Check if there is any duplicate...

duplicate_count = train["Id"].nunique() - train.shape[0];

if(duplicate_count == 0):

    print("There is no duplicate data in the train data.")

else:

    print("There are %d duplicate data in the train data." %duplicate_count)

train.drop("Id", axis = 1, inplace = True)
sns.distplot(train['SalePrice'], kde=False)
med = train['SalePrice'].median()

medAbsDev = abs(train['SalePrice'] - med).median()

# expected value of MAD = 0.6745 * sigma

ind = (abs(train['SalePrice'] - med)*.6745 / medAbsDev) > 3.5

print("There are suspected %d outliers in the train data." % sum(ind))

train = train[-ind]

# removing further data as suggested by the author of the dataset

# https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

train = train[train.GrLivArea < 4000]
sns.heatmap(train.isnull(), yticklabels=False, cbar=False)
train["Alley"].fillna("None", inplace=True) # NA = "no alley access"

train["BsmtQual"].fillna("None", inplace=True) # NA = "No Basement"

train["BsmtCond"].fillna("None", inplace=True) # NA = "No Basement"

train["BsmtExposure"].fillna("None", inplace=True) # NA = "No Basement"

train["BsmtFinType1"].fillna("None", inplace=True) # NA = "No Basement"

train["BsmtFinSF1"].fillna(0, inplace=True) # NA = "No Basement"

train["BsmtFinType2"].fillna("None", inplace=True) # NA = "No Basement"

train["BsmtFinSF2"].fillna(0, inplace=True) # NA = "No Basement"

train["BsmtFullBath"].fillna(0, inplace=True) # NA = "No Basement"

train["BsmtHalfBath"].fillna(0, inplace=True) # NA = "No Basement"

train["BsmtUnfSF"].fillna(0, inplace=True) # NA = "No Basement"

train["TotalBsmtSF"].fillna(0, inplace=True) # NA = "No Basement"

train["KitchenQual"].fillna(0, inplace=True)

train["Functional"].fillna(0, inplace=True)

train["FireplaceQu"].fillna("None", inplace=True) # NA = "No Fireplace"

train["GarageType"].fillna("None", inplace=True) # NA = "No Garage"

train["GarageQual"].fillna(0, inplace=True) # NA = "No Garage"

train["GarageCond"].fillna(0, inplace=True) # NA = "No Garage"

train["GarageYrBlt"].fillna("None", inplace=True) # NA = "No Garage"

train["GarageFinish"].fillna("None", inplace=True) # NA = "No Garage"

train["GarageCars"].fillna(0, inplace=True) # NA = "No Garage"

train["GarageArea"].fillna(0, inplace=True) # NA = "No Garage"

train["PoolQC"].fillna("None", inplace=True) # NA = "No Pool"

train["Fence"].fillna("None", inplace=True) # NA = "No Fence"

train["MiscFeature"].fillna("None", inplace=True) # NA = "No MiscFeature"



# not clearly described

train["LotFrontage"].fillna(0, inplace=True)

train["MasVnrType"].fillna("None", inplace=True)

train["MasVnrArea"].fillna(0, inplace=True)

train["Electrical"].fillna("SBrkr", inplace=True)

train["Utilities"].fillna("None", inplace=True)



# no missing data

# MSZoning, LotArea, Street, LotShape, LandContour, Utilities, LotConfig, LandSlope, Neighborhood, Condition1

# Condition2, BldgType, HouseStyle, OverallQual, OverallCond, YearBuilt, YearRemodAdd, RoofStyle, RoofMatl

# Exterior1st, Exterior2nd, ExterQual, ExterCond, Foundation, Heating, HeatingQC, CentralAir, 1stFlrSF, 2ndFlrSF

# LowQualFinSF, GrLivArea, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr

# TotRmsAbvGrd, Fireplaces, GarageCars, GarageArea, PavedDrive, WoodDeckSF, OpenPorchSF, EnclosedPorch

# 3SsnPorch, ScreenPorch, PoolArea, MiscVal, MoSold, YrSold, SaleType, SaleCondition
# convert suitable numerical columns to categorical columns

train = train.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 50 : "SC50", 60 : "SC60",

    70 : "SC70", 75 : "SC75", 80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 150 : "SC150", 160 : "SC160",

    180 : "SC180", 190 : "SC190"},

"MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sep",

    10 : "Oct", 11 : "Nov", 12 : "Dec"} })



# convert suitable categorical columns to numerical columns

train = train.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

"BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

"BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

"BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

"BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

"ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

"ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

"FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},

"GarageCond" : {"None" : 0, "No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"GarageQual" : {"None" : 0, "No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

"LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

"PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

"PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

"Street" : {"Grvl" : 1, "Pave" : 2},

"Utilities" : {"None" : 0, "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}} )
# Grade

train["OverallGrade"] = train["OverallQual"] * train["OverallCond"]

train["GarageGrade"] = train["GarageQual"] * train["GarageCond"]

train["ExterGrade"] = train["ExterQual"] * train["ExterCond"]



# Score

train["KitchenScore"] = train["KitchenAbvGr"] * train["KitchenQual"]

train["FireplaceScore"] = train["Fireplaces"] * train["FireplaceQu"]

train["GarageScore"] = train["GarageArea"] * train["GarageQual"]

train["PoolScore"] = train["PoolArea"] * train["PoolQC"]



# Aggregate

train["TotalBath"] = train["BsmtFullBath"] + train["BsmtHalfBath"]/2 + train["FullBath"] + train["HalfBath"]/2

train["AllSF"] = train["GrLivArea"] + train["TotalBsmtSF"]

train["AllFlrsSF"] = train["1stFlrSF"] + train["2ndFlrSF"]

train["AllPorchSF"] = train["OpenPorchSF"] + train["EnclosedPorch"] + train["3SsnPorch"] + train["ScreenPorch"]
corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

corr.SalePrice
train["AllSF-s2"] = train["AllSF"] ** 2

#train["AllSF-s3"] = train["AllSF"] ** 3

train["AllSF-Sq"] = np.sqrt(train["AllSF"])

train["OverallQual-s2"] = train["OverallQual"] ** 2

#train["OverallQual-s3"] = train["OverallQual"] ** 3

train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])

train["AllFlrsSF-s2"] = train["AllFlrsSF"] ** 2

#train["AllFlrsSF-s3"] = train["AllFlrsSF"] ** 3

train["AllFlrsSF-Sq"] = np.sqrt(train["AllFlrsSF"])

train["GrLivArea-s2"] = train["GrLivArea"] ** 2

#train["GrLivArea-s3"] = train["GrLivArea"] ** 3

train["GrLivArea-Sq"] = np.sqrt(train["GrLivArea"])

train["ExterQual-s2"] = train["ExterQual"] ** 2

#train["ExterQual-s3"] = train["ExterQual"] ** 3

train["ExterQual-Sq"] = np.sqrt(train["ExterQual"])
categorical_features = train.select_dtypes(include = ["object"]).columns

numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

train_num = train[numerical_features]

train_cat = train[categorical_features]



from scipy.stats import skew

skewness = train_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print("%d out of %d numerical features are skewed. Apply log transform to these features." % (skewness.shape[0],

    train_num.shape[1]))

train_num.loc[:,skewness.index] = np.log1p(np.asarray(train_num[skewness.index] , dtype=float))



# Apply one-hot encoding on columns of categorical features

train_cat = pd.get_dummies(train_cat)
y = np.log1p(train["SalePrice"])

X_train, X_test, y_train, y_test = train_test_split(pd.concat([train_num, train_cat], axis = 1),

                                                    y, test_size = 0.3, random_state = 514)

# apply fitting only on training data

stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])
# Define function of error measure to be root-mean-square error. Apply cross-validation using the error function.

from sklearn.metrics import mean_squared_error, make_scorer

scorer = make_scorer(mean_squared_error, greater_is_better = True)



def rmse_cv_train(model):

    rmse = np.sqrt(cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse = np.sqrt(cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)
from sklearn.linear_model import LinearRegression

# Linear Regression

lr = LinearRegression()

lr.fit(X_train, y_train)

print("Mean RMSE on Training set :", rmse_cv_train(lr).mean()) # to-do: check why surprisingly high values may come up

print("Mean RMSE on Test set :", rmse_cv_test(lr).mean())

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(X_test)



# Plot residuals (using code by Julien)

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training House Prices")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation House Prices")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training House Prices")

plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation House Prices")

plt.title("Linear regression")

plt.xlabel("Predicted House Prices")

plt.ylabel("Real House Prices")

plt.legend(loc = "upper left")

plt.show()
# roughly search alpha and l1_ratio

from sklearn.linear_model import ElasticNetCV

elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

                          alphas = [0.0001, 0.001, 0.01, 0.1, 1], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )
print("Try again for more precision with l1_ratio centered around " + str(ratio))

elasticNet = ElasticNetCV(l1_ratio = [ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1],

                          alphas = [0.0001, 0.001, 0.01, 0.1, 1], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(X_train, y_train)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )
print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

      " and alpha centered around " + str(alpha))

elasticNet = ElasticNetCV(l1_ratio = ratio,

                          alphas = [alpha * .6, alpha * .7, alpha * .8, alpha * .9, 

                                    alpha, alpha * 1.1, alpha * 1.2, alpha * 1.3, alpha * 1.4], 

                          max_iter = 50000, cv = 5)

elasticNet.fit(X_train, y_train)

print("Best l1_ratio :", elasticNet.l1_ratio_)

print("Best alpha :", elasticNet.alpha_ )



print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())

print("ElasticNet RMSE on Test set :", rmse_cv_test(elasticNet).mean())

y_train_ela = elasticNet.predict(X_train)

y_test_ela = elasticNet.predict(X_test)
# Plot residuals

plt.scatter(y_train_ela, y_train_ela - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_ela, y_test_ela - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train, y_train_ela, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test, y_test_ela, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with ElasticNet regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(elasticNet.coef_, index = X_train.columns)

print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the ElasticNet Model")

plt.show()
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")



test["Alley"].fillna("None", inplace=True) # NA = "no alley access"

test["BsmtQual"].fillna("None", inplace=True) # NA = "No Basement"

test["BsmtCond"].fillna("None", inplace=True) # NA = "No Basement"

test["BsmtExposure"].fillna("None", inplace=True) # NA = "No Basement"

test["BsmtFinType1"].fillna("None", inplace=True) # NA = "No Basement"

test["BsmtFinSF1"].fillna(0, inplace=True) # NA = "No Basement"

test["BsmtFinType2"].fillna("None", inplace=True) # NA = "No Basement"

test["BsmtFinSF2"].fillna(0, inplace=True) # NA = "No Basement"

test["BsmtFullBath"].fillna(0, inplace=True) # NA = "No Basement"

test["BsmtHalfBath"].fillna(0, inplace=True) # NA = "No Basement"

test["BsmtUnfSF"].fillna(0, inplace=True) # NA = "No Basement"

test["TotalBsmtSF"].fillna(0, inplace=True) # NA = "No Basement"

test["KitchenQual"].fillna(0, inplace=True)

test["Functional"].fillna(0, inplace=True)

test["FireplaceQu"].fillna("None", inplace=True) # NA = "No Fireplace"

test["GarageType"].fillna("None", inplace=True) # NA = "No Garage"

test["GarageQual"].fillna(0, inplace=True) # NA = "No Garage"

test["GarageCond"].fillna(0, inplace=True) # NA = "No Garage"

test["GarageYrBlt"].fillna("None", inplace=True) # NA = "No Garage"

test["GarageFinish"].fillna("None", inplace=True) # NA = "No Garage"

test["GarageCars"].fillna(0, inplace=True) # NA = "No Garage"

test["GarageArea"].fillna(0, inplace=True) # NA = "No Garage"

test["PoolQC"].fillna("None", inplace=True) # NA = "No Pool"

test["Fence"].fillna("None", inplace=True) # NA = "No Fence"

test["MiscFeature"].fillna("None", inplace=True) # NA = "No MiscFeature"



test["LotFrontage"].fillna(0, inplace=True)

test["MasVnrType"].fillna("None", inplace=True)

test["MasVnrArea"].fillna(0, inplace=True)

test["Electrical"].fillna("SBrkr", inplace=True)

test["Utilities"].fillna("None", inplace=True)



test = test.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 50 : "SC50", 60 : "SC60",

    70 : "SC70", 75 : "SC75", 80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 150 : "SC150", 160 : "SC160",

    180 : "SC180", 190 : "SC190"},

"MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun", 7 : "Jul", 8 : "Aug", 9 : "Sep",

    10 : "Oct", 11 : "Nov", 12 : "Dec"} })



test = test.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

"BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

"BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

"BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

"BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

"ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

"ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

"FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},

"GarageCond" : {"None" : 0, "No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"GarageQual" : {"None" : 0, "No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

"LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

"LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

"PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

"PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

"Street" : {"Grvl" : 1, "Pave" : 2},

"Utilities" : {"None" : 0, "ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}} )



test["OverallGrade"] = test["OverallQual"] * test["OverallCond"]

test["GarageGrade"] = test["GarageQual"] * test["GarageCond"]

test["ExterGrade"] = test["ExterQual"] * test["ExterCond"]



test["KitchenScore"] = test["KitchenAbvGr"] * test["KitchenQual"]

test["FireplaceScore"] = test["Fireplaces"] * test["FireplaceQu"]

test["GarageScore"] = test["GarageArea"] * test["GarageQual"]

test["PoolScore"] = test["PoolArea"] * test["PoolQC"]



test["TotalBath"] = test["BsmtFullBath"] + test["BsmtHalfBath"]/2 + test["FullBath"] + test["HalfBath"]/2

test["AllSF"] = test["GrLivArea"] + test["TotalBsmtSF"]

test["AllFlrsSF"] = test["1stFlrSF"] + test["2ndFlrSF"]

test["AllPorchSF"] = test["OpenPorchSF"] + test["EnclosedPorch"] + test["3SsnPorch"] + test["ScreenPorch"]



test["AllSF-s2"] = test["AllSF"] ** 2

#test["AllSF-s3"] = test["AllSF"] ** 3

test["AllSF-Sq"] = np.sqrt(test["AllSF"])

test["OverallQual-s2"] = test["OverallQual"] ** 2

#test["OverallQual-s3"] = test["OverallQual"] ** 3

test["OverallQual-Sq"] = np.sqrt(test["OverallQual"])

test["AllFlrsSF-s2"] = test["AllFlrsSF"] ** 2

#test["AllFlrsSF-s3"] = test["AllFlrsSF"] ** 3

test["AllFlrsSF-Sq"] = np.sqrt(test["AllFlrsSF"])

test["GrLivArea-s2"] = test["GrLivArea"] ** 2

#test["GrLivArea-s3"] = test["GrLivArea"] ** 3

test["GrLivArea-Sq"] = np.sqrt(test["GrLivArea"])

test["ExterQual-s2"] = test["ExterQual"] ** 2

#test["ExterQual-s3"] = test["ExterQual"] ** 3

test["ExterQual-Sq"] = np.sqrt(test["ExterQual"])



test_num = test[numerical_features]

test_cat = test[categorical_features]

test_num.loc[:,skewness.index] = np.log1p(np.asarray(test_num[skewness.index] , dtype=float))

test_cat = pd.get_dummies(test_cat)

test_processed = pd.concat([test_num, test_cat], axis = 1)

test_processed.loc[:, numerical_features] = stdSc.transform(test_processed.loc[:, numerical_features])
final_train, final_test = X_train.align(test_processed, join='left', axis=1, fill_value=0)

y_test = elasticNet.predict(final_test)

sub = pd.DataFrame()

sub['Id'] = test["Id"]

sub['SalePrice'] = np.expm1(y_test)

sub.to_csv('submission.csv',index=False)