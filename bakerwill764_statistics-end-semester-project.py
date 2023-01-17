import pandas as pd

import matplotlib.pyplot as plt

from statsmodels.compat import lzip

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

import seaborn as sns

import numpy as np

from IPython.display import display

from scipy.stats import norm, shapiro, skew, boxcox

from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

sns.set(font_scale=1)
train = pd.read_csv('../input/train.csv')

train.shape

pd.set_option('display.max_rows', 20)

pd.set_option('display.max_columns', 20)

train
# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")

plt.title("SalePrice vs GrLivArea")

plt.xlabel("GrLivArea (sqft)")

plt.ylabel("SalePrice ($)")

plt.show()



train = train[train.GrLivArea < 4000]
sns.distplot(train['SalePrice']);
# Log transform on target to make more normal

#train.SalePrice = np.log1p(train.SalePrice)



# Box-cox transform on target to make more normal

train.SalePrice = train.SalePrice + 1

train.SalePrice = boxcox(train.SalePrice)[0]
#histogram

sns.distplot(train['SalePrice']);
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
# Replace NA with 'None' for relevant columns

train["Alley"].fillna('No', inplace=True)

filter = train["Alley"] == 'None'

train["Alley"].where(filter, inplace=True)

train["Alley"].fillna('Yes', inplace=True)



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
# Some numerical features are actually really categories

train = train.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                      })
#missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(5)
train["Electrical"].describe().top
#dealing with missing data

train = train.drop((missing_data[missing_data['Total'] > 1]).index,1)

train.loc[:, "Electrical"] = train.loc[:, "Electrical"].fillna("SBrkr")

y = train.SalePrice

train.isnull().sum().max() #just checking that there's no missing data
# Clean some more columns

train = train.drop(columns="Id")



train = train.drop(columns="PoolQC")

# Change PoolArea to yes/no "Pool"

train = train.rename(columns={"PoolArea": "Pool"})

train["Pool"] = train["Pool"].replace(0, "No")

filter = train["Pool"] == "No"

train["Pool"].where(filter, inplace=True)

train["Pool"].fillna("Yes", inplace=True)



# Drop columns due to single observations

train = train.drop(columns="Utilities") #NoSeWa only one besides AllPub
# Encode some categorical features as ordered numbers when there is information in the order

train = train.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "Neighborhood" : {"MeadowV" : 1, "IDOTRR" : 1, "BrDale" : 1, "OldTown" : 1, "Edwards" : 1, 

                                         "BrkSide" : 1, "Sawyer" : 1, "Blueste" : 1, "SWISU" : 1, "NAmes" : 1, 

                                         "NPkVill" : 1, "Mitchel" : 1, "SawyerW" : 2, "Gilbert" : 2, "NWAmes" : 2, 

                                         "Blmngtn" : 2, "CollgCr" : 2, "ClearCr" : 2, "Crawfor" : 2, "Veenker" : 3, 

                                         "Somerst" : 3, "Timber" : 3, "StoneBr" : 3, "NoRidge" : 3, "NridgHt" : 3},

                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4}

                       })

train["Neighborhood"] = pd.to_numeric(train["Neighborhood"])
# Find most important features relative to target

print("Find most important features relative to target")

corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.set(font_scale=1)

sns.heatmap(corrmat, vmax=.8, square=True);
#zoomed saleprice correlation matrix

k = 18 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1)

f, ax = plt.subplots(figsize=(9, 7))

hm = sns.heatmap(cm, vmax=0.8, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Differentiate numerical features (minus the target) and categorical features

categorical_features = train.select_dtypes(include = ["object"]).columns

numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

train_num = train[numerical_features]

train_cat = train[categorical_features]
# Log transform of the skewed numerical features to lessen impact of outliers

# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed

skewness1 = train_num.apply(lambda x: skew(x))

skewness1 = skewness1[abs(skewness1) > 0.5]

print(str(skewness1.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness1.index

print(skewness1)

train_num[skewed_features] = np.log1p(train_num[skewed_features])
skewness2 = train_num.apply(lambda x: skew(x))

skewness2 = skewness2[abs(skewness2) > 0.5]

print(str(skewness2.shape[0]) + " skewed numerical features remain:")

print(skewness2)
# Create dummy features for categorical values via one-hot encoding

train_cat = pd.get_dummies(train_cat)
# Join categorical and numerical features

from statsmodels.tools.tools import add_constant



train = pd.concat([train_num, train_cat], axis = 1)

print("New number of features : " + str(train.shape[1]))



y_train = y

X_train = train

X_train_ols = add_constant(X_train)



# Partition the dataset in train + validation sets

#X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3)

#print("X_train : " + str(X_train.shape))

#print("X_test : " + str(X_test.shape))

#print("y_train : " + str(y_train.shape))

#print("y_test : " + str(y_test.shape))
# Drop variables based on strong correlation

X_train_ols = X_train_ols.drop(columns="GarageArea") # less correlated than GarageCars

X_train_ols = X_train_ols.drop(columns="TotRmsAbvGrd") # less correlated than GrLivArea

X_train_ols = X_train_ols.drop(columns="1stFlrSF") # less correlated than TotalBsmtSF#X_train_ols = X_train_ols.drop(columns="Fireplaces") # Less correlated than FireplaceQu

# Drop perfectly collinear variables

X_train_ols = X_train_ols.drop(columns="BsmtFinSF1")

X_train_ols = X_train_ols.drop(columns="BsmtFinSF2")
# VIF Calculation

from statsmodels.stats.outliers_influence import variance_inflation_factor

X_train_ols["Intercept"] = 1

vif = pd.DataFrame()

vif["variables"] = X_train_ols.columns

vif["VIF"] = [variance_inflation_factor(X_train_ols.values, i) for i in range(X_train_ols.shape[1])]

vif = vif.replace([np.inf, -np.inf], np.nan)

vif = vif.dropna()

pd.set_option('display.max_rows', None)

print(vif.sort_values("VIF", ascending=False).head(15))
# Drop variables based on VIF

# Re-calculate VIF after each removal, removing highest VIF in the next step

X_train_ols = X_train_ols.drop(columns="MiscVal")

X_train_ols = X_train_ols.drop(columns="MasVnrArea")

X_train_ols = X_train_ols.drop(columns="2ndFlrSF")

X_train_ols = X_train_ols.drop(columns="GarageCond")

X_train_ols = X_train_ols.drop(columns="YearBuilt")

X_train_ols = X_train_ols.drop(columns="GarageQual")

X_train_ols = X_train_ols.drop(columns="BsmtQual")

X_train_ols = X_train_ols.drop(columns="BsmtCond")

X_train_ols = X_train_ols.drop(columns="FireplaceQu")
# StatsModels OLS MLR Model

from statsmodels.regression.linear_model import OLS

sm_model = OLS(y_train,X_train_ols)

results = sm_model.fit()

print(results.summary())
from statsmodels.stats.stattools import jarque_bera

name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']

test = jarque_bera(results.resid)

lzip(name, test)
from scipy.stats import shapiro

name = ['W', 'P-value']

test = shapiro(results.resid)

lzip(name, test)
from statsmodels.graphics.gofplots import qqplot

fig = qqplot(results.resid)

plt.title("Q-Q Plot")

plt.show()
# Standardize numerical features

stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

#X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])
# 2* Ridge

ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 

                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,

                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 

                cv = 10)

ridge.fit(X_train, y_train)

alpha = ridge.alpha_

print("Best alpha :", alpha)



y_train_rdg = ridge.predict(X_train)

#y_test_rdg = ridge.predict(X_test)

print("R^2: ", ridge.score(X_train, y_train))

adj_r2 = 1 - (1 - ridge.score(X_train, y_train))*((1455)/(1455-252-1))

print("Adjusted R^2: ", adj_r2)



# Plot residuals

plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")

#plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

#plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 8.5, xmax = 10, color = "red")

plt.show()



# Plot predictions

#plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")

#plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")

#plt.title("Linear regression with Ridge regularization")

#plt.xlabel("Predicted values")

#plt.ylabel("Real values")

#plt.legend(loc = "upper left")

#plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

#plt.show()



# Plot important coefficients

coefs = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Model")

plt.show()
fig = qqplot(y_train_rdg - y_train)

plt.show()
# 3* Lasso

lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 

                          0.3, 0.6, 1], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



print("Try again for more precision with alphas centered around " + str(alpha))

lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 

                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 

                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 

                          alpha * 1.4], 

                max_iter = 50000, cv = 10)

lasso.fit(X_train, y_train)

alpha = lasso.alpha_

print("Best alpha :", alpha)



y_train_las = lasso.predict(X_train)

#y_test_las = lasso.predict(X_test)

print("R^2: ", lasso.score(X_train, y_train))

adj_r2 = 1 - (1 - lasso.score(X_train, y_train))*((1455)/(1455-252-1))

print("Adjusted R^2: ", adj_r2)

#print("Test R^2: ", lasso.score(X_test, y_test))



# Plot residuals

plt.scatter(y_train_las, y_train_las - y_train)

#plt.scatter(y_test_las, y_test_las - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Lasso regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

#plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 8.5, xmax = 10, color = "red")

plt.show()



# Plot predictions

#plt.scatter(y_train_las, y_train, c = "blue", marker = "s", label = "Training data")

#plt.scatter(y_test_las, y_test, c = "lightgreen", marker = "s", label = "Validation data")

#plt.title("Linear regression with Lasso regularization")

#plt.xlabel("Predicted values")

#plt.ylabel("Real values")

#plt.legend(loc = "upper left")

#plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

#plt.show()



# Plot important coefficients

coefs = pd.Series(lasso.coef_, index = X_train.columns)

print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Lasso Model")

plt.show()
fig = qqplot(y_train_las - y_train)

plt.show()
sns.distplot(resid)
name = ['W', 'P-value']

test = shapiro(resid)

lzip(name, test)