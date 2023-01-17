# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import xgboost as xgb

import lightgbm as lgb

from scipy import stats

from scipy.stats import norm, skew

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.linear_model import BayesianRidge, LassoLarsIC

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone



import warnings

warnings.filterwarnings("ignore")



%matplotlib inline
# get data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

# sample = pd.read_csv("../input/sample_submission.csv")

print("train.csv shape: " + str(train.shape))

print("test.csv shape: " + str(test.shape))

# print("sample.csv shape: " + str(sample.shape))
train.head()
train.SalePrice.describe()
# get a overview of the SalePrice distribution

sns.distplot(train.SalePrice);
# look at some general outliers

plt.scatter(train.GrLivArea, train.SalePrice, c="blue", s=2)

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()
# remove outliers, i.e. GrLivArea > 4000

train = train.drop(train[(train.GrLivArea > 4000) & (train.SalePrice < 300000)].index)
plt.scatter(train.GrLivArea, train.SalePrice, c="blue", s=2);

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()
# log transform the SalePrice, so that the bigger values does not have a to big impact

# on the smaller ones

#train = train.drop(train.loc[train.Electrical.isnull()].index)

# save the "Id" column

train_ID = train["Id"]

test_ID = test["Id"]



# drop the "Id" column

train.drop("Id", axis=1, inplace=True)

test.drop("Id", axis=1, inplace=True)
# build a correlation matrix to get an idea of the important or relevant categories

corrmat = train.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, square=True);
cols = corrmat.nlargest(15, "SalePrice")["SalePrice"].index

values = corrmat.nlargest(15, "SalePrice")["SalePrice"].values

high_corrmat = np.corrcoef(train[cols].values.T)

f, ax = plt.subplots(figsize=(10,8))

sns.set(font_scale=1.25)

high_heatmap = sns.heatmap(high_corrmat, cbar=True, annot=True, fmt=".2f",

                          annot_kws={"size": 10}, square=True,

                          yticklabels=cols.values, xticklabels=cols.values)

high_cor = pd.concat([pd.Series(cols), pd.Series(values)], keys=["index", "value"], axis=1)

print(high_cor)
train.SalePrice = np.log1p(train.SalePrice)
ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))
# find the missing data

total_na = all_data.isnull().sum()

total_na = total_na.drop(total_na[total_na == 0].index).sort_values(ascending=False)

missing = pd.DataFrame({"Total Missing data": total_na})

missing.head(20)
# handel missing values for features where median/mean or most common value does not

# make sense



# Alley: NA for Alley means "no alley access"

all_data.loc[:, "Alley"] = all_data.loc[:, "Alley"].fillna("NoAl")



# BedroomAbvGr: NA for Bedrooms above ground means 0 Bedrooms

all_data.loc[:, "BedroomAbvGr"] = all_data.loc[:, "BedroomAbvGr"].fillna(0)



# BsmtXXX: NA for basement features means there is "no basement"

all_data.loc[:, "BsmtQual"] = all_data.loc[:, "BsmtQual"].fillna("NoBa")

all_data.loc[:, "BsmtCond"] = all_data.loc[:, "BsmtCond"].fillna("NoBa")

all_data.loc[:, "BsmtExposure"] = all_data.loc[:, "BsmtExposure"].fillna("NoBa")

all_data.loc[:, "BsmtFinType1"] = all_data.loc[:, "BsmtFinType1"].fillna("NoBa")

all_data.loc[:, "BsmtFinType2"] = all_data.loc[:, "BsmtFinType2"].fillna("NoBa")



# Electrical: NA means no electricity

# Electrical: Should be dropped

#train = train.drop(train.loc[train.Electrical.isnull()].index)

all_data.loc[:, "Electrical"] = all_data.loc[:, "Electrical"].fillna("NoEL")



# Fence: NA means "no fence"

all_data.loc[:, "Fence"] = all_data.loc[:, "Fence"].fillna("NoFe")



# FireplaceQu: data description says NA means "no fireplace"

all_data.loc[:, "FireplaceQu"] = all_data.loc[:, "FireplaceQu"].fillna("NoFi")



# GarageType etc: data description says NA for garage features is "no garage"

all_data.loc[:, "GarageType"] = all_data.loc[:, "GarageType"].fillna("NoGa")

all_data.loc[:, "GarageFinish"] = all_data.loc[:, "GarageFinish"].fillna("NoGa")

all_data.loc[:, "GarageQual"] = all_data.loc[:, "GarageQual"].fillna("NoGa")

all_data.loc[:, "GarageCond"] = all_data.loc[:, "GarageCond"].fillna("NoGa")

# use for GarageYrBlt the average

#train.loc[:, "GarageArea"] = train.loc[:, "GarageArea"].fillna(0)

#train.loc[:, "GarageCars"] = train.loc[:, "GarageCars"].fillna(0)



#

# LotFrontage : NA most likely means no lot frontage

# to much data missing, try mean()

#train.loc[:, "LotFrontage"] = train.loc[:, "LotFrontage"].fillna(0)

#



# MasVnrType: NA means no veneer

all_data.loc[:, "MasVnrType"] = all_data.loc[:, "MasVnrType"].fillna("None")

all_data.loc[:, "MasVnrArea"] = all_data.loc[:, "MasVnrArea"].fillna(0)



# MiscFeature: NA means "no misc feature"

all_data.loc[:, "MiscFeature"] = all_data.loc[:, "MiscFeature"].fillna("NoFe")

#train.loc[:, "MiscVal"] = train.loc[:, "MiscVal"].fillna(0)



# PoolQC: NA means "no pool"

all_data.loc[:, "PoolQC"] = all_data.loc[:, "PoolQC"].fillna("NoPo")

#train.loc[:, "PoolArea"] = train.loc[:, "PoolArea"].fillna(0)
#Some numerical features are actually really categories

all_data = all_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr",

                                   5 : "May", 6 : "Jun", 7 : "Jul", 8 : "Aug",

                                   9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"},

                       

                      })

# all_data.MSSubClass = all_data.MSSubClass.apply(str)

all_data.OverallCond = all_data.OverallCond.astype(str)

# all_data.MoSold = all_data.MoSold.astype(str)

all_data.YrSold = all_data.YrSold.astype(str)
# encode some categorical features as ordered numbers when there is information in

# the order

all_data = all_data.replace({#"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "BsmtCond": {"NoBa": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4,

                                   "Ex": 5},

                       "BsmtExposure": {"NoBa": 0, "Mn": 1, "Av": 2, "Gd": 3},

                       "BsmtFinType1" : {"NoBa" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3,

                                        "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"NoBa" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3,

                                        "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"NoBa" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4,

                                    "Ex" : 5},

                       

                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       # try both

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       

                       "GarageCond" : {"NoGa" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "GarageQual" : {"NoGa" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       # try both

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       # try both

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       

                       #"PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       

                       "PoolQC" : {"NoPo" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       

                       #"Street" : {"Grvl" : 1, "Pave" : 2},

                       # try both

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}

                            })
# differentiate numerical and categorical features

cat_features = all_data.select_dtypes(include=["object"]).columns

num_features = all_data.select_dtypes(exclude=["object"]).columns

#num_features = num_features.drop("SalePrice")

print("Numerical features: " + str(len(num_features)))

print("Categorical features: " + str(len(cat_features)))

all_data_num = all_data[num_features]

all_data_cat = all_data[cat_features]
print("NAs for numerical features in train : " + str(all_data_num.isnull().values.sum()))

all_data_num = all_data_num.fillna(all_data_num.mean())

print("Remaining NAs for numerical features in train : " + str(all_data_num.isnull().values.sum()))
# log transform of the skewed numerical features to lessen impact the outliers

skewness = all_data_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness.index

all_data_num[skewed_features] = np.log1p(all_data_num[skewed_features])
# create dummy features for categorical values via one-hot encoding

print("NAs for categorical features in train : " + str(all_data_cat.isnull().values.sum()))

all_data_cat = pd.get_dummies(all_data_cat)

print("Remaining NAs for categorical features in train : " + str(all_data_cat.isnull().values.sum()))
all_data = pd.get_dummies(all_data)

print(all_data.shape)
train = all_data[:ntrain]

test = all_data[ntrain:]
# join numerical and categorical features

all_data = pd.concat([all_data_num, all_data_cat], axis=1)

print("New number of features : " + str(all_data.shape[1]))

train = all_data[:ntrain]

test = all_data[ntrain:]

# # Partition the dataset in train + validation sets

# X_train, X_test, y_train, y_test = train_test_split(train.values, y_train, test_size = 0.3, random_state = 0)

# print("X_train : " + str(X_train.shape))

# print("X_test : " + str(X_test.shape))

# print("y_train : " + str(y_train.shape))

# print("y_test : " + str(y_test.shape))

print(len(train))

print(len(y_train))
# stdSc = StandardScaler()

# train.values.loc[:, num_features] = stdSc.fit_transform(train.values.loc[:, num_features])

# X_test.loc[:, num_features] = stdSc.transform(X_test.loc[:, num_features])
scorer = make_scorer(mean_squared_error, greater_is_better=False)



def rmse_cv_train(model):

    rmse = np.sqrt(-cross_val_score(model, train.values, y_train, scoring=scorer, cv=10))

    return(rmse)
l1 = [0.00001, 0.00003, 0.00006, 0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60, 100]

l2 = np.arange(100) + 1

l3 = l2 * 0.1

l4 = l2 * 0.01

l5 = l2 * 0.001

l6 = l2 * 0.0001

l7 = l2 * 0.00001

l8 = l2 * 0.000001

print(l3)
l2_1 = np.arange(1000) + 1

l3_1 = l2_1 * 0.1

l4_1 = l2_1 * 0.01

ridge = RidgeCV(alphas = [5.63])

ridge.fit(train.values, y_train)

alpha = ridge.alpha_

print("Best alpha: ", alpha)

print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())
lasso = LassoCV(alphas = [0.00031], max_iter=50000, cv=10)

lasso.fit(train.values, y_train)

# alpha = lasso.alpha_

# print("Best alpha: ", alpha)

print("Lasso RMSE on Training set: ", rmse_cv_train(lasso).mean())
# 4* ElasticNet

elasticNet = ElasticNetCV(l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 

                                    0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(train.values, y_train)

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Try again for more precision with l1_ratio centered around " + str(ratio))

elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],

                          alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(train.values, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("Now try again for more precision on alpha, with l1_ratio fixed at " + str(ratio) + 

      " and alpha centered around " + str(alpha))

elasticNet = ElasticNetCV(l1_ratio = ratio,

                          alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, alpha * .9, 

                                    alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, 

                                    alpha * 1.35, alpha * 1.4], 

                          max_iter = 50000, cv = 10)

elasticNet.fit(train.values, y_train)

if (elasticNet.l1_ratio_ > 1):

    elasticNet.l1_ratio_ = 1    

alpha = elasticNet.alpha_

ratio = elasticNet.l1_ratio_

print("Best l1_ratio :", ratio)

print("Best alpha :", alpha )



print("ElasticNet RMSE on Training set :", rmse_cv_train(elasticNet).mean())

y_train_ela = elasticNet.predict(train.values)
KRR = KernelRidge(alpha = [600], kernel="polynomial", degree=1.94, coef0=40)

KRR.fit(train.values, y_train)

print("Kernel Ridge Regression RMSE on Training set :", rmse_cv_train(KRR).mean())
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)

GBoost.fit(train.values, y_train)

print("XGB RMSE on Training set: ", rmse_cv_train(GBoost).mean())
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # define clones of ther original models to fit the data

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # train all cloned models

        for model in self.models_:

            model.fit(X, y)

            

        return self

    

    # predictions for cloned models

    def predict(self, X):

        predictions = np.column_stack([model.predict(X) for model in self.models_])

        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (lasso, elasticNet, KRR, GBoost))



print("Averaged Models RMSE on Training set :", rmse_cv_train(averaged_models).mean())

# averaged2 = AveragingModels(models = (lasso, KRR, GBoost))

# print("Averaged Models RMSE on Training set: ", rmse_cv_train(averaged2).mean())
# averaged3 = AveragingModels(models = (elasticNet, KRR, GBoost))

# print("Averaged Models RMSE on Training set: ", rmse_cv_train(averaged3).mean())

# averaged4 = AveragingModels(models = (KRR, GBoost))

# print("Averaged Models RMSE on Training set: ", rmse_cv_train(averaged4).mean())

# averaged5 = AveragingModels(models = (lasso, GBoost))

# print("Averaged Models RMSE on Training set: ", rmse_cv_train(averaged5).mean())
def rmse(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
averaged_models.fit(train.values, y_train)

averaged_train_pred = averaged_models.predict(train.values)

averaged_pred = np.expm1(averaged_models.predict(test.values))

print(rmse(y_train, averaged_train_pred))
sub = pd.DataFrame()

sub["Id"] = test_ID

sub["SalePrice"] = averaged_pred

sub.to_csv("submission2.csv", index=False)
# averaged5.fit(train.values, y_train)

# averaged5_train_pred = averaged5.predict(train.values)

# averaged5_pred = np.expm1(averaged5.predict(test.values))

# print(rmse(y_train, averaged5_train_pred))
# sub = pd.DataFrame()

# sub["Id"] = test_ID

# sub["SalePrice"] = averaged5_pred

# sub.to_csv("submission.csv", index=False)