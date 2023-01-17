#import libraries 

import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt 

import seaborn as sns

from scipy import stats
#load data 

trainset = pd.read_csv("../input/train.csv")
trainset.head(5)
trainset.info()
testset = pd.read_csv("../input/test.csv")
testset.head(5)
ids = testset["Id"]
trainset = trainset.drop("Id", axis = 1)

testset = testset.drop("Id", axis = 1)
plt.scatter(x = trainset["GrLivArea"], y = trainset["SalePrice"])

plt.ylabel("SalePrice")

plt.xlabel("GrLivArea")

plt.show()
#removing the outiliers which are more than 4000 sq ft from the dataset 

traindata = trainset.drop(trainset[(trainset["GrLivArea"]>4000) & (trainset["SalePrice"]<300000)].index).reset_index(drop=True)
trainset_len = len(traindata)

trainset_len
traindata["SalePrice"].describe()
#finding the relationship between the saleprice and other variables 

correlation = traindata.corr()

correlation.sort_values(["SalePrice"])

correlation.SalePrice
from scipy.stats import skew, kurtosis
skew = skew(traindata["SalePrice"])

kurt = kurtosis(traindata["SalePrice"])
print("skewness:", skew)

print("kurtosis:", kurt)
mean = trainset["SalePrice"].describe()['mean']

std = trainset["SalePrice"].describe()['std']

#note standard deviation is the deviation of the data from the mean 
print("mean:", mean)

print("standard deviation:", std)
#plotting skewness 

sns.distplot(trainset["SalePrice"], norm_hist = True)

plt.axvline(x = mean, color = "red", linestyle = "--", label = "mean")

plt.legend()

plt.grid()

plt.show()
Transformed = np.log1p(traindata["SalePrice"])

Transformed.describe()
mean_Trans = Transformed.describe()["mean"]

std_Trans = Transformed.describe()["std"]
print("mean:", mean_Trans)

print("standard deviation:", std_Trans)
print("skewness of transformed:", Transformed.skew())

print("kurtosis of transformed:", Transformed.kurt())
sns.distplot(Transformed, norm_hist = True)

plt.axvline(x = mean_Trans, color = "red", linestyle = "--", label = "mean")

plt.legend()

plt.grid()

plt.show()
#concating the trainset and the testset 

data = pd.concat((traindata, testset))
data.head()
#drop saleprice since it the dependent variable and does not appear in the test set

dataa = data.drop("SalePrice", axis = 1)
#identifying missing values 

dataa.isnull().values.any()
#how many missing values are there in total 

dataa.isnull().sum().sum()
#which variables contain the missing values and their total

dataa.isnull().sum()
pd.set_option('display.max_rows', 79) #show all variables 

dataa.isnull().sum()
#For Lotfrontage, the median can be used to replace the missing values since it is a float 

median = dataa["LotFrontage"].median()

dataa["LotFrontage"].fillna(median, inplace = True)

#all missing values have been replaced by the median 
#missing values for alley. The mssing values in alley shows that there is no access to alley and since it is a 

#categorical variable, we ca replace the missing values with None

dataa["Alley"].fillna("None", inplace = True)
#missing values here can be filled with none meaning no Masonry veneer type and also it is a categorical data 

dataa["MasVnrType"].fillna("None", inplace = True)
#missing values here can be filled with none meaning no Masonry veneer type and also it is a quantiative data 

#the number of missing values for MasVnrType is not the same as the number of missing values for MasVnrArea. Therefore,

#where the value of MasVnrType is avaliable, the median of MasVnrArea would be used to impute the missing value 

dataa.loc[dataa["MasVnrType"] == "None","MasVnrArea"] = 0

dataa["MasVnrArea"] = dataa["MasVnrArea"].fillna(dataa["MasVnrArea"].median())
#the missing values in BsmtQual shows that there is no basement which means they can be replaced with None since 

#the variable is a qualitative data. same can be done for BsmtCond, BsmtExposure, BsmtFinType2, BsmtFinType1

dataa["BsmtQual"].fillna("None", inplace = True)

dataa["BsmtCond"].fillna("None", inplace = True)

dataa["BsmtExposure"].fillna("None", inplace = True)

dataa["BsmtFinType1"].fillna("None", inplace = True)

dataa["BsmtFinType2"].fillna("None", inplace = True)
#the missing values in BsmtFinSF1 shows that there is no basement which means they can be replaced with 0 since

#it is a quantitative variable. Same can be done to BsmtFinSF2, BsmtFullBath, BsmtHalfBath, BsmtUnfSF and TotalBsmtSF

dataa["BsmtFinSF1"].fillna(0, inplace = True)

dataa["BsmtFinSF2"].fillna(0, inplace = True)

dataa["BsmtFullBath"].fillna(0, inplace = True)

dataa["BsmtHalfBath"].fillna(0, inplace = True)

dataa["BsmtUnfSF"].fillna(0, inplace = True)

dataa["TotalBsmtSF"].fillna(0, inplace = True)
dataa["Exterior1st"].mode()
#setting the missing value to Vinyl Siding since that is what is mostly used 

dataa["Exterior1st"].fillna("VinylSd", inplace = True)
dataa["Exterior2nd"].mode()
#setting the missing value to Vinyl Siding since that is what is mostly used 

dataa["Exterior2nd"].fillna("VinylSd", inplace = True)
dataa["Functional"].mode()
#setting the missing value to Typical Functionality since that is what is mostly used 

dataa["Functional"].fillna("Typ", inplace = True)
#the missing values for FireplaceQu shows that there is non fireplace and since it is also a categorical data, NA can

#be replaced with None 

dataa["FireplaceQu"].fillna("None", inplace = True)
dataa["GarageType"].fillna("None", inplace = True)
dataa.loc[dataa["GarageType"] == "None","GarageYrBlt"] = dataa["YearBuilt"][dataa["GarageType"]=="None"]

dataa.loc[dataa["GarageType"] == "None","GarageCars"] = 0

dataa.loc[dataa["GarageType"] == "None","GarageArea"] = 0

dataa.loc[dataa["GarageType"] == "None","GarageFinish"] = "None"

dataa.loc[dataa["GarageType"] == "None","GarageQual"] = "None"

dataa.loc[dataa["GarageType"] == "None","GarageCond"] = "None"
dataa["GarageArea"] = dataa["GarageArea"].fillna(dataa["GarageArea"].median())

dataa["GarageCars"] = dataa["GarageCars"].fillna(dataa["GarageCars"].median())

dataa["GarageYrBlt"] = dataa["GarageYrBlt"].fillna(dataa["GarageYrBlt"].median())
dataa["GarageFinish"].mode()
dataa["GarageFinish"].fillna("Unf", inplace = True)
dataa["GarageQual"].mode()
dataa["GarageQual"].fillna("TA", inplace = True)
dataa["GarageCond"].mode()
dataa["GarageCond"].fillna("TA", inplace = True)
dataa["Electrical"].mode()
#setting the missing value to Standard Circuit Breakers since that is what is mostly used 

dataa["Electrical"].fillna("SBrkr", inplace = True)
#missing values for Fence means no fence, there NA can be replaced with none since it is a categorical data 

dataa["Fence"].fillna("None", inplace = True)
dataa["KitchenQual"].mode()
#setting the missing value to Typical/Average since that is what is mostly used 

dataa["KitchenQual"].fillna("TA", inplace = True)
dataa["MSZoning"].mode()
#setting the missing value to Residential Low Density since that is what is mostly used 

dataa["MSZoning"].fillna("RL", inplace = True)
#missing values for PoolQC means no pool, there NA can be replaced with none since it is a categorical data 

dataa["PoolQC"].fillna("None", inplace = True)
#missing values for MiscFeature means no fence, there NA can be replaced with none since it is a categorical data 

dataa["MiscFeature"].fillna("None", inplace = True)
dataa["SaleType"].mode()
#setting the missing value to Warranty Deed - Conventional since that is what is mostly used 

dataa["SaleType"].fillna("WD", inplace = True)
dataa["Utilities"].mode()
#setting the missing value to All public Utilities (E,G,W,& S) since that is what is mostly used 

dataa["Utilities"].fillna("AllPub", inplace = True)
#checking to see if there are still missing values 

dataa.isnull().values.any()
dataa["Area"] = dataa["TotalBsmtSF"] + dataa["GrLivArea"]
dataa["TotalFlrSF"] = dataa["1stFlrSF"] + dataa["2ndFlrSF"]
dataa["AllPorchSF"] = dataa["OpenPorchSF"] + dataa["EnclosedPorch"] + dataa["3SsnPorch"] + dataa["ScreenPorch"]
dataa["Utilities"].value_counts()
dataa = dataa.drop("Utilities", axis = 1)
# converting variable which are to be categorical data instead of numerical into categorical 

#note, year is a categorical data since the ratio between two years is not meaning hence classified as categorical



for i in ("MSSubClass", "MoSold", "YrSold"):

    dataa[i] = dataa[i].astype(str)

dataa.dtypes
numerical_features = dataa.dtypes[dataa.dtypes != "object"].index 

skewed_feats = dataa[numerical_features].skew().sort_values(ascending=False)

skewness = pd.DataFrame({'Skewness' :skewed_feats})

skewness
numerical_features = ["MiscVal", "PoolArea", "LotArea", "LowQualFinSF", "3SsnPorch", "KitchenAbvGr", "EnclosedPorch",

                      "ScreenPorch", "BsmtHalfBath", "MasVnrArea", "OpenPorchSF", "WoodDeckSF",

                      "LotFrontage", "Area", "GrLivArea", "TotalFlrSF", "1stFlrSF", "AllPorchSF", "BsmtFinSF2"]
for i in numerical_features:

    dataa[i] = np.log1p(dataa[i])

dataa.head()
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')

for i in cols:

    le = LabelEncoder()

    dataa[i] = le.fit_transform(list(dataa[i]))  
dataa = pd.get_dummies(dataa)
y = traindata["SalePrice"]

x_train = dataa[:trainset_len]

x_test = dataa[trainset_len:]
from math import sqrt

from sklearn.model_selection import cross_val_score
def rmse_md(estimator, x, y):

    rmse = np.sqrt(-cross_val_score(estimator, x, y, cv= 15, scoring="neg_mean_squared_error")).mean()

    return rmse  #Root mean square error
#in order to deal with the overall outliers in the dataset

from sklearn.preprocessing import RobustScaler
R = RobustScaler()

R.fit(x_train)

x_train = R.transform(x_train) #normilaze the dataset to withstand outliers 
R.fit(x_test)

x_test = R.transform(x_test)
y = Transformed #log transform of the target variable
from sklearn import linear_model
br = linear_model.BayesianRidge()

br.fit(x_train, y)

score = rmse_md(br, x_train, y)

print("score of Bayesian Ridge :",score)

R_squared = br.score(x_train, y)

print("score of R_squared :", R_squared)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 720).fit(x_train, y)

score = rmse_md(rfr, x_train, y)

print("score of Random Forest :",score)

R_squared = rfr.score(x_train, y)

print("score of R_squared :", R_squared)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()

gbr.fit(x_train, y)

score = rmse_md(gbr, x_train, y)

print("score of Gradient Boosting :",score)

R_squared = gbr.score(x_train, y)

print("score of R_squared :", R_squared)
from sklearn.linear_model import RidgeCV  #using the cross validation 
rg = RidgeCV(cv = 15)

rg.fit(x_train, y)

score = rmse_md(rg, x_train, y)

print("score of Ridge Regression :", score)

R_squared = rg.score(x_train, y)

print("score of R_squared :", R_squared)
from sklearn.linear_model import LassoCV #using the cross validation
ls = LassoCV(cv = 15)

ls.fit(x_train, y)

score = rmse_md(ls, x_train, y)

print("score of Lasso Regression :", score)

R_squared = ls.score(x_train, y)

print("score of R_squared :", R_squared)
from sklearn.linear_model import ElasticNetCV #using the cross validation
en = ElasticNetCV(cv = 15)

en.fit(x_train, y)

score = rmse_md(en, x_train, y)

print("score of ElasticNet Regression :", score)

R_squared = en.score(x_train, y)

print("score of R_squared :", R_squared)
import xgboost 
xgb = xgboost.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=7200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

xgb.fit(x_train, y)

score = rmse_md(xgb, x_train, y)

print("score of xgb:", score)

R_squared = xgb.score(x_train, y)

print("score of R_squared :", R_squared)
import lightgbm 
lgb = lightgbm.LGBMRegressor(objective='regression',num_leaves= 5,

                              learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8,

                              bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9,

                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

lgb.fit(x_train, y)

score = rmse_md(lgb, x_train, y)

print("score of lgb:", score)

R_squared = lgb.score(x_train, y)

print("score of R_squared :", R_squared)
y_pred_bayesian = np.expm1(br.predict(x_test))

y_pred_lassocv = np.expm1(ls.predict(x_test))

y_pred_GradientBoost = np.expm1(gbr.predict(x_test))

y_pred_elasticnetcv = np.expm1(en.predict(x_test))

y_pred_ridge = np.expm1(rg.predict(x_test))

y_pred_random = np.expm1(rfr.predict(x_test))

y_pred_xgb = np.expm1(xgb.predict(x_test))

y_pred_lgb = np.expm1(lgb.predict(x_test))
print("score of br :",  rmse_md(br, x_train, y))

print("score of ls:", rmse_md(ls, x_train, y))

print("score of en:", rmse_md(en, x_train, y))

print("score of rg:", rmse_md(rg, x_train, y))

print("score of gbr:", rmse_md(gbr, x_train, y))

print("score of xgb:", rmse_md(xgb, x_train, y)) 

print("score of lgb:", rmse_md(lgb, x_train, y))
ensemble_score = (0.15 *rmse_md(br, x_train, y) +

                  0.17 * rmse_md(ls, x_train, y) +

                  0.09 * rmse_md(gbr, x_train, y) +

                  0.17 * rmse_md(en, x_train, y) + 

                  0.15 * rmse_md(rg, x_train, y) + 

                  0.13 * rmse_md(xgb, x_train, y) + 

                  0.14 * rmse_md(lgb, x_train, y))

ensemble_score 
output =(0.15 * y_pred_bayesian +

         0.17 * y_pred_lassocv + 

         0.09 * y_pred_GradientBoost + 

         0.17 * y_pred_elasticnetcv  + 

         0.15* y_pred_ridge +

         0.13 * y_pred_xgb + 

         0.14 * y_pred_lgb)
SalePredict = pd.DataFrame(output)

SalePredict["Id"] = ids

SalePredict = SalePredict.rename(columns={0: "SalePrice"})

SalePredict = SalePredict[["Id","SalePrice"]]

SalePredict.to_csv("Submission.csv", index=False)