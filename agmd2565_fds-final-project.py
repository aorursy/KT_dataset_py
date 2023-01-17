# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV

from sklearn.metrics import mean_squared_error, make_scorer,r2_score

from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

from scipy.stats import skew

import xgboost as xgb

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns

from statistics import mode

# Definitions

pd.set_option('display.float_format', lambda x: '%.3f' % x)

from IPython import get_ipython

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



# Any results you write to the current directory are saved as output.

# Get data

train = pd.read_csv("../input/train.csv")

#test = pd.read_csv(os.path.join('data', 'test.csv'))

test = pd.read_csv("../input/test.csv")

print("train : " + str(train.shape))

SP=train.SalePrice

print(train.columns)

print(train.head())

print(train['SalePrice'].describe())



print("test : " + str(train.shape))

print(test.columns)



# Looking for outliers, as indicated in https://ww2.amstat.org/publications/jse/v19n3/decock.pdf

plt.scatter(train.GrLivArea, train.SalePrice, c = "blue", marker = "s")

plt.title("Looking for outliers")

plt.xlabel("GrLivArea")

plt.ylabel("SalePrice")

plt.show()



#scatter plot totalbsmtsf/saleprice

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



#scatter plot 'YearBuilt'/saleprice

var = 'YearBuilt'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));



#scatter plot 'OverallQual'/saleprice

var = 'OverallQual'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=90);



#scatter plot 'Neighborhood'/saleprice

plt.figure(figsize = (12, 6))

sns.countplot(x = 'Neighborhood', data = train)

xt = plt.xticks(rotation=45)

var = 'Neighborhood'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=45);



#scatter plot 'SaleCondition'/saleprice

plt.figure(figsize = (12, 6))

sns.countplot(x = 'SaleCondition', data = train)

xt = plt.xticks(rotation=45)

var = 'SaleCondition'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

f, ax = plt.subplots(figsize=(16, 8))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);

plt.xticks(rotation=45);



#Heating

pd.crosstab(train.HeatingQC, train.CentralAir)

sns.factorplot('HeatingQC', 'SalePrice', hue = 'CentralAir', estimator = np.mean, data = train, 

             size = 4.5, aspect = 1.4)



#Electrical

fig, ax = plt.subplots(1, 2, figsize = (10, 4))

sns.boxplot('Electrical', 'SalePrice', data = train, ax = ax[0]).set(ylim = (0, 400000))

sns.countplot('Electrical', data = train)

plt.tight_layout()



#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);



cor_dict = corrmat['SalePrice'].to_dict()

del cor_dict['SalePrice']

print("List the numerical features decendingly by their correlation with Sale Price:\n")

for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):

    print("{0}: \t{1}".format(*ele))
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()

data_sorted = corrmat.sort_values(['SalePrice'], ascending=True)

print(data_sorted[['SalePrice']].head(10))



plt.figure(figsize=(18,8))

futures=cols[1:]

for i in range(6):

    ii = '23'+str(i+1)

    plt.subplot(ii)

    feature = futures[i]

    plt.scatter(train[feature], train['SalePrice'], facecolors='none',edgecolors='k',s = 75)

    sns.regplot(x = feature, y = 'SalePrice', data = train,scatter=False, color = 'Blue')

    ax=plt.gca() 

    ax.set_ylim([0,800000])
#TRAIN + test

# %% concatenate test and train data 

# ( Dont include SalePrice since this is target variable)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                                test.loc[:,'MSSubClass':'SaleCondition']))



#missing data

total = all_data.isnull().sum().sort_values(ascending=True)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(total)



# Handle missing values for features where median/mean or most common value doesn't make sense

# Alley : data description says NA means "no alley access"

all_data.loc[:, "Alley"] = all_data.loc[:, "Alley"].fillna("None")

all_data.loc[:, "BedroomAbvGr"] = all_data.loc[:, "BedroomAbvGr"].fillna(0)

# BsmtQual etc : data description says NA for basement features is "no basement"

all_data.loc[:, "BsmtQual"] = all_data.loc[:, "BsmtQual"].fillna("No")

all_data.loc[:, "BsmtCond"] = all_data.loc[:, "BsmtCond"].fillna("No")

all_data.loc[:, "BsmtExposure"] = all_data.loc[:, "BsmtExposure"].fillna("No")

all_data.loc[:, "BsmtFinType1"] = all_data.loc[:, "BsmtFinType1"].fillna("No")

all_data.loc[:, "BsmtFinType2"] = all_data.loc[:, "BsmtFinType2"].fillna("No")

all_data.loc[:, "BsmtFullBath"] = all_data.loc[:, "BsmtFullBath"].fillna(0)

all_data.loc[:, "BsmtHalfBath"] = all_data.loc[:, "BsmtHalfBath"].fillna(0)

all_data.loc[:, "BsmtUnfSF"] = all_data.loc[:, "BsmtUnfSF"].fillna(0)



# CentralAir : NA most likely means No

all_data.loc[:, "CentralAir"] = all_data.loc[:, "CentralAir"].fillna("N")



# Condition : NA most likely means Normal

all_data.loc[:, "Condition1"] = all_data.loc[:, "Condition1"].fillna("Norm")

all_data.loc[:, "Condition2"] = all_data.loc[:, "Condition2"].fillna("Norm")



# EnclosedPorch : NA most likely means no enclosed porch

all_data.loc[:, "EnclosedPorch"] = all_data.loc[:, "EnclosedPorch"].fillna(0)



# External stuff : NA most likely means average

all_data.loc[:, "ExterCond"] = all_data.loc[:, "ExterCond"].fillna("TA")

all_data.loc[:, "ExterQual"] = all_data.loc[:, "ExterQual"].fillna("TA")



all_data.loc[:, "Fireplaces"] = all_data.loc[:, "Fireplaces"].fillna(0)

# Functional : data description says NA means typical

all_data.loc[:, "Functional"] = all_data.loc[:, "Functional"].fillna("Typ")

# GarageType etc : data description says NA for garage features is "no garage"

all_data.loc[:, "GarageType"] = all_data.loc[:, "GarageType"].fillna("No")

all_data.loc[:, "GarageFinish"] = all_data.loc[:, "GarageFinish"].fillna("No")

all_data.loc[:, "GarageQual"] = all_data.loc[:, "GarageQual"].fillna("No")

all_data.loc[:, "GarageCond"] = all_data.loc[:, "GarageCond"].fillna("No")

all_data.loc[:, "GarageArea"] = all_data.loc[:, "GarageArea"].fillna(0)

all_data.loc[:, "GarageCars"] = all_data.loc[:, "GarageCars"].fillna(0)

# HalfBath : NA most likely means no half baths above grade

all_data.loc[:, "HalfBath"] = all_data.loc[:, "HalfBath"].fillna(0)

# HeatingQC : NA most likely means typical

all_data.loc[:, "HeatingQC"] = all_data.loc[:, "HeatingQC"].fillna("TA")

# KitchenAbvGr : NA most likely means 0

all_data.loc[:, "KitchenAbvGr"] = all_data.loc[:, "KitchenAbvGr"].fillna(0)

# KitchenQual : NA most likely means typical

all_data.loc[:, "KitchenQual"] = all_data.loc[:, "KitchenQual"].fillna("TA")





# MasVnrType : NA most likely means no veneer

all_data.loc[:, "MasVnrType"] = all_data.loc[:, "MasVnrType"].fillna("None")

all_data.loc[:, "MasVnrArea"] = all_data.loc[:, "MasVnrArea"].fillna(0)



all_data.loc[:, "MiscVal"] = all_data.loc[:, "MiscVal"].fillna(0)

# OpenPorchSF : NA most likely means no open porch

all_data.loc[:, "OpenPorchSF"] = all_data.loc[:, "OpenPorchSF"].fillna(0)

# PavedDrive : NA most likely means not paved

all_data.loc[:, "PavedDrive"] = all_data.loc[:, "PavedDrive"].fillna("N")

# PoolQC : data description says NA means "no pool"



all_data.loc[:, "PoolArea"] = all_data.loc[:, "PoolArea"].fillna(0)

# SaleCondition : NA most likely means normal sale

all_data.loc[:, "SaleCondition"] = all_data.loc[:, "SaleCondition"].fillna("Normal")

# ScreenPorch : NA most likely means no screen porch

all_data.loc[:, "ScreenPorch"] = all_data.loc[:, "ScreenPorch"].fillna(0)

# TotRmsAbvGrd : NA most likely means 0

all_data.loc[:, "TotRmsAbvGrd"] = all_data.loc[:, "TotRmsAbvGrd"].fillna(0)

# Utilities : NA most likely means all public utilities

all_data.loc[:, "Utilities"] = all_data.loc[:, "Utilities"].fillna("AllPub")

# WoodDeckSF : NA most likely means no wood deck

all_data.loc[:, "WoodDeckSF"] = all_data.loc[:, "WoodDeckSF"].fillna(0)

# Fence : data description says NA means "no fence"

all_data.loc[:, "Fence"] = all_data.loc[:, "Fence"].fillna("No")

# FireplaceQu : data description says NA means "no fireplace"

all_data.loc[:, "FireplaceQu"] = all_data.loc[:, "FireplaceQu"].fillna("No")

# PoolQC : data description says NA means "no pool"

all_data.loc[:, "PoolQC"] = all_data.loc[:, "PoolQC"].fillna("No")

all_data.loc[:, "Exterior2nd"] = all_data.loc[:, "Exterior2nd"].fillna("Other")

all_data.loc[:, "TotalBsmtSF"] = all_data.loc[:, "TotalBsmtSF"].fillna("Floor")

all_data.loc[:, "Electrical"] = all_data.loc[:, "Electrical"].fillna("AllPub")

all_data.loc[:, "MSZoning"] = all_data.loc[:, "MSZoning"].fillna("RL") 

# LotFrontage : NA most likely means no lot frontage

all_data.loc[:, "LotFrontage"] = all_data.loc[:, "LotFrontage"].fillna(0)

all_data.loc[:, "Exterior1st"] = all_data.loc[:, "Exterior1st"].fillna('Other')

all_data.loc[:, "BsmtFinSF2"] = all_data.loc[:, "BsmtFinSF2"].fillna('Unf')

all_data.loc[:, "BsmtFinSF1"] = all_data.loc[:, "BsmtFinSF1"].fillna(0)

all_data.loc[:, "SaleType"] = all_data.loc[:, "SaleType"].fillna('Oth')

total = all_data.isnull().sum().sort_values(ascending=True)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(total)
# 2* Combinations of existing features

f, (ax1, ax2, ax3) = sns.plt.subplots(1, 3)

data_total = pd.concat([train['SalePrice'], train['TotalBsmtSF']], axis=1)

data_total.plot.scatter(x='TotalBsmtSF', y='SalePrice', ylim=(0, 800000), ax=ax1)

data1 = pd.concat([train['SalePrice'], train['1stFlrSF']], axis=1)

data1.plot.scatter(x='1stFlrSF', y='SalePrice', ylim=(0, 800000), ax=ax2)

data2 = pd.concat([train['SalePrice'], train['2ndFlrSF']], axis=1)

data2.plot.scatter(x='2ndFlrSF', y='SalePrice', ylim=(0, 800000), ax=ax3)

sns.plt.show()

# all of them are the area information of the house and their relationships with SalePrice are similar, so our strategy is creating a new feature named as TotalSF to add those three features, then drop them.



# new feature named as TotalBath to add those three features, then drop them.



all_data['TotalBath'] = all_data["FullBath"] + (0.5 * all_data["HalfBath"])



# new feature named GarageGrade- quality of the house



all_data = all_data.replace({"GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                           "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                         }

                        )

all_data['GarageGrade'] = all_data['GarageQual'] * all_data['GarageCond']



# Overall quality of the exterior

# External stuff : NA most likely means average



all_data = all_data.replace({"ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},

                       }

                     )

all_data['ExterGrade'] = all_data['ExterQual'] * all_data['ExterCond']



# Create new features

# 1* Simplifications of existing features

# Simplified overall fireplace score

# FireplaceQu : data description says NA means "no fireplace"



all_data = all_data.replace({"FireplaceQu" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       }

                     )

all_data["SimplFireplaceQu"] = all_data.FireplaceQu.replace({1 : 1, # bad

                                                       2 : 1, 3 : 1, # average

                                                       4 : 2, 5 : 2 # good

                                                      })

all_data["SimplFireplaceScore"] = all_data["Fireplaces"] * all_data["SimplFireplaceQu"]



# Create new features

# 1* Simplifications of existing features

all_data["SimplOverallQual"] = all_data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad

                                                       4 : 2, 5 : 2, 6 : 2, # average

                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good

                                                      })

all_data["SimplOverallCond"] = all_data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad

                                                       4 : 2, 5 : 2, 6 : 2, # average

                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good

                                                      })

all_data["SimplFunctional"] = all_data.Functional.replace({1 : 1, 2 : 1, # bad

                                                     3 : 2, 4 : 2, # major

                                                     5 : 3, 6 : 3, 7 : 3, # minor

                                                     8 : 4 # typical

                                                    })

all_data["SimplKitchenQual"] = all_data.KitchenQual.replace({1 : 1, # bad

                                                       2 : 1, 3 : 1, # average

                                                       4 : 2, 5 : 2 # good

                                                      })

all_data["SimplExterCond"] = all_data.ExterCond.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

all_data["SimplExterQual"] = all_data.ExterQual.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

#Generate features related to time. 

#For example, we generate a "New_House" column by considering if the house was built and sold in the same year

all_data["New_House"] = (all_data["YearRemodAdd"] == all_data["YrSold"]) * 1
# Some numerical features are actually really categories

all_data = all_data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                      })
all_data = all_data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},

                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 

                                         "ALQ" : 5, "GLQ" : 6},

                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},

                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 

                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},

                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},

                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},

                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},

                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}

                     )

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

categorical_feats = all_data.dtypes[all_data.dtypes == "object"].index

print(categorical_feats)

                    
all_data.drop(['MiscFeature', 'Fence', 'PoolQC', 'FireplaceQu', 'Alley'], axis=1, inplace=True)

all_data.drop(['GarageYrBlt','GarageArea'], axis=1, inplace=True)

all_data = all_data[all_data.GrLivArea < 4000]

# Differentiate numerical features (minus the target) and categorical features in train

categorical = all_data.select_dtypes(include = ["object"]).columns

numerical = all_data.select_dtypes(exclude = ["object"]).columns

# Create dummy features for categorical values via one-hot encoding

print("NAs for features in all_data : " + str(all_data.isnull().values.sum()))

all_data_cat = pd.get_dummies(all_data[categorical])

all_data_num=all_data[numerical]

all_data = pd.concat([all_data_num, all_data_cat], axis = 1)

print("New number of features : " + str(all_data.shape[1]))
# 3* Polynomials for existing features

#cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

#print(cols)

#cols=cols[1:]

#for col in cols:

  #all_data[col+'-2'] = all_data.loc[:,col]**2

  #all_data[col+'-3'] = all_data.loc[:,col]**3

  #all_data[col+'-sqrt'] = np.sqrt(np.absolute(all_data.loc[:,col]))
train = all_data[:train.shape[0]]

train['SalePrice'] = SP

print("train : " + str(train.shape))



test = all_data[train.shape[0]:]

print("train : " + str(train.shape))



# Log transform the target for official scoring

train.SalePrice = np.log1p(train.SalePrice)

y = train.SalePrice



#total = train.isnull().sum().sort_values(ascending=True)

#print(total)

#print("test : " + str(train.shape))

#total = test.isnull().sum().sort_values(ascending=True)

#print(total)
# Find most important features relative to target

print("Find most important features relative to target")

corr = train.corr()

corr.sort_values(["SalePrice"], ascending = False, inplace = True)

print(corr.SalePrice)

# Differentiate numerical features (minus the target) and categorical features in train

categorical_features = train.select_dtypes(include = ["object"]).columns

numerical_features = train.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features train : " + str(len(numerical_features)))

print("Categorical features train : " + str(len(categorical_features)))

train_num = train[numerical_features]

train_num = train_num.fillna(train_num.median())

train_cat = train[categorical_features]



# Differentiate numerical features (minus the target) and categorical features in test

categorical_features1 = test.select_dtypes(include = ["object"]).columns

numerical_features1 = test.select_dtypes(exclude = ["object"]).columns

print("Numerical features test: " + str(len(numerical_features1)))

print("Categorical features test: " + str(len(categorical_features1)))

test_num = test[numerical_features1]

test_cat = test[categorical_features1]

test_num = test_num.fillna(test_num.median())

print("Na in test")

total = test.isnull().sum().sort_values(ascending=True)

print(total)

# Log transform of the skewed numerical features to lessen impact of outliers

# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed

skewness = train_num.apply(lambda x: skew(x))

#skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + "train skewed numerical features to log transform")

skewed_features = skewness.index

train_num[skewed_features] = np.log1p(train_num[skewed_features])

#test

skewness = test_num.apply(lambda x: skew(x))

#skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " test skewed numerical features to log transform")

skewed_features = skewness.index

test_num[skewed_features] = np.log1p(test_num[skewed_features])



total = test.isnull().sum().sort_values(ascending=True)

print(total)
# Join train categorical and numerical features

# Create dummy features for categorical values via one-hot encoding

#print("NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))

#train_cat = pd.get_dummies(train_cat)

#print("Remaining NAs for categorical features in train : " + str(train_cat.isnull().values.sum()))

#train = pd.concat([train_num, train_cat], axis = 1)

#print("New number of features : " + str(train.shape[1]))



# Join test categorical and numerical features

# Create dummy features for categorical values via one-hot encoding

#print("NAs for categorical features in test : " + str(test_cat.isnull().values.sum()))

#test_cat = pd.get_dummies(test_cat)

#print("Remaining NAs for categorical features in test : " + str(test_cat.isnull().values.sum()))

#test = pd.concat([test_num, test_cat], axis = 1)

#print("New number of features test: " + str(test.shape[1]))



# Partition the dataset in train + validation sets

X_train, X_test, y_train, y_test = train_test_split(train, y, test_size = 0.3, random_state = 0)

print("X_train : " + str(X_train.shape))

print("X_test : " + str(X_test.shape))

print("y_train : " + str(y_train.shape))

print("y_test : " + str(y_test.shape))



# Standardize numerical features

#fit() : used for generating learning model parameters from training data

#transform() : parameters generated from fit() method,applied upon model to generate transformed data set.

#fit_transform() : combination of fit() and transform() api on same data set

stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])

X_test.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])

# Define error measure for official scoring : RMSE

scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, X_test, y_test, scoring = scorer, cv = 10))

    return(rmse)
# Linear Regression

# Create linear regression object

regr = LinearRegression()

regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)

y_test_pred = regr.predict(X_test)

# The coefficients

print('Coefficients: \n', regr.coef_)

# The mean squared error

print("RMSE on Training set :", rmse_cv_train(regr).mean())

print("RMSE on Test set :", rmse_cv_test(regr).mean())

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score( y_test, y_test_pred))

# Plot residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()

#predictions = regr.predict(test)

#predictions = np.expm1(predictions)

#print(predictions)

#predictions = pd.DataFrame(predictions, columns=['SalePrice'])

#predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

#predictions.to_csv('predictios.csv', sep=",", index = False)
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



print("Ridge RMSE on Training set :", rmse_cv_train(ridge).mean())

print("Ridge RMSE on Test set :", rmse_cv_test(ridge).mean())

y_train_rdg = ridge.predict(X_train)

y_test_rdg = ridge.predict(X_test)



# Plot residuals

plt.scatter(y_train_rdg, y_train_rdg - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test_rdg - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_rdg, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_rdg, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression with Ridge regularization")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()



# Plot important coefficients

coefs = pd.Series(ridge.coef_, index = X_train.columns)

print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \

      str(sum(coefs == 0)) + " features")

imp_coefs = pd.concat([coefs.sort_values().head(10),

                     coefs.sort_values().tail(10)])

imp_coefs.plot(kind = "barh")

plt.title("Coefficients in the Ridge Model")

plt.show()



predictions = regr.predict(test)

predictions = np.expm1(predictions)

print(predictions)

predictions = pd.DataFrame(predictions, columns=['SalePrice'])

predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

predictions.to_csv('predictios.csv', sep=",", index = False)
 # Fitting the model and predicting using xgboost

regr = xgb.XGBRegressor(colsample_bytree=0.4,

                        gamma=0.045,learning_rate=0.07,max_depth=20,min_child_weight=1.5,

                        n_estimators=300,

                        reg_alpha=0.65,

                        reg_lambda=0.45,

                        subsample=0.95)

regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)

y_test_pred = regr.predict(X_test)

# The mean squared error

print("RMSE on Training set :", rmse_cv_train(regr).mean())

print("RMSE on Test set :", rmse_cv_test(regr).mean())

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score( y_test, y_test_pred))

predictions = regr.predict(test)

predictions = np.expm1(predictions)

predictions = pd.DataFrame(predictions, columns=['SalePrice'])

predictions = pd.concat((test.iloc[:, 0], predictions), axis = 1)

predictions.to_csv('C:\\backup\\FUNDAMENTAL OF DATA SCIENCE\\predictios.csv', sep=",", index = False)
# Plot residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Xgboost")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Xgboost")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
regr = RandomForestRegressor(10, max_features='sqrt')

regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)

y_test_pred = regr.predict(X_test)

# The mean squared error

print("RMSE on Training set :", rmse_cv_train(regr).mean())

print("RMSE on Test set :", rmse_cv_test(regr).mean())

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score( y_test, y_test_pred))
# Plot residuals

plt.scatter(y_train_pred, y_train_pred - y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("RandomForestRegressor")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("RandomForestRegressor")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
regr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,max_depth=1, 

                                 random_state=0, loss='ls').fit(X_train, y_train)

regr.fit(X_train, y_train)

y_train_pred = regr.predict(X_train)

y_test_pred = regr.predict(X_test)

# The mean squared error

print("RMSE on Training set :", rmse_cv_train(regr).mean())

print("RMSE on Test set :", rmse_cv_test(regr).mean())

# Explained variance score: 1 is perfect prediction

print('Variance score: %.2f' % r2_score( y_test, y_test_pred))
# compute test set deviance

test_score = np.zeros(100, dtype=np.float64)

for i, y_pred in enumerate(regr.staged_predict(X_test)):

    test_score[i] = regr.loss_(y_test, y_pred)



plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.title('Deviance')

plt.plot(np.arange(100) + 1, regr.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(100) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')
