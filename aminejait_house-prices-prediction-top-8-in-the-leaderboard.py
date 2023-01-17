import pandas as pd

import numpy as np

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import Ridge, ElasticNet, Lasso

from sklearn.kernel_ridge import KernelRidge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, make_scorer

from scipy.stats import skew

from IPython.display import display

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
#Importing both the trainand test files as pandas dataframes.

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

print('assert the length of the dataframes:'+ str(len(test)) + str(len(train)))
#removing all GrLivArea bigger than 4000 (outliers). Recommended by the creator of the dataset.



train = train[train.GrLivArea < 4000]

print('the lenght of train after removing >4000 sqf lots:'+ str(len(train)))



#Create the y vector out of the SalePrice new log1plus distribution

y = np.log1p(train.SalePrice)

#Store the Id of the test df in a variable (testId) for later use. Concatenate the test and train dfs for feature engineering.

testId = test.Id

data= pd.concat((train, test), axis=0, sort= False).reset_index(drop=True )

print(data.shape)

mtrain = train.shape[0]

mtest = test.shape[0]

print(mtrain, mtest)
# Check for duplicates

idsUnique = len(set(data.Id))

idsTotal = data.shape[0]

idsDupli = idsTotal - idsUnique

print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")



# Drop Id column

data.drop("Id", axis = 1, inplace = True)

print('the shape of data:' + str(data.shape))

print(data.columns)
#handling missing values for categorical data. Pandas profiling helped check for missing values 



# Alley : data description says NA means "no alley access"

data.loc[:, "Alley"] = data.loc[:, "Alley"].fillna("None")

# BedroomAbvGr : NA most likely means 0

data.loc[:, "BedroomAbvGr"] = data.loc[:, "BedroomAbvGr"].fillna(0)

# Bsmt : data description says NA for basement features is "no basement"

data.loc[:, "BsmtQual"] = data.loc[:, "BsmtQual"].fillna("No")

data.loc[:, "BsmtCond"] = data.loc[:, "BsmtCond"].fillna("No")

data.loc[:, "BsmtExposure"] = data.loc[:, "BsmtExposure"].fillna("No")

data.loc[:, "BsmtFinType1"] = data.loc[:, "BsmtFinType1"].fillna("No")

data.loc[:, "BsmtFinType2"] = data.loc[:, "BsmtFinType2"].fillna("No")

data.loc[:, "BsmtFullBath"] = data.loc[:, "BsmtFullBath"].fillna(0)

data.loc[:, "BsmtHalfBath"] = data.loc[:, "BsmtHalfBath"].fillna(0)

data.loc[:, "BsmtUnfSF"] = data.loc[:, "BsmtUnfSF"].fillna(0)

#no missing vals from centralair

#no missing vals from conditio1 and condition2

#no missing vals in Enclosed porch only 0s

# Fence : data description says NA means "no fence"

data.loc[:, "Fence"] = data.loc[:, "Fence"].fillna("No")

#fireplace 0 means 0 squarespace, and fireplacequ NA means no fireplace

data.loc[:, "FireplaceQu"] = data.loc[:, "FireplaceQu"].fillna("No")

#nothing missing in functional

#Garage missing values

data.loc[:, "GarageCond"] = data.loc[:, "GarageCond"].fillna("No")

data.loc[:, "GarageQual"] = data.loc[:, "GarageQual"].fillna("No")

data.loc[:, "GarageFinish"] = data.loc[:, "GarageFinish"].fillna("No")

data.loc[:, "GarageType"] = data.loc[:, "GarageType"].fillna("No")

# LotFrontage : NA most likely means no lot frontage

data.loc[:, "LotFrontage"] = data.loc[:, "LotFrontage"].fillna(0)

data.loc[:, "MiscFeature"] = data.loc[:, "MiscFeature"].fillna("None")

# PoolQC : data description says NA means "no pool"

data.loc[:, "PoolQC"] = data.loc[:, "PoolQC"].fillna("No")

# Some numerical features are actually really categories

data = data.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 

                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 

                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 

                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},

                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",

                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}

                      })
# Encode some categorical features as ordered numbers when there is information in the order

data = data.replace({"Alley" : {"Grvl" : 1, "Pave" : 2},

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

                       "PoolQC" : {"No" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},

                       "Street" : {"Grvl" : 1, "Pave" : 2},

                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4},

                       "MSZoning": {"A" : 1, "C" : 2, "FV": 3, "I" : 4, "RH" : 5, "RL" : 6, "RP" : 7, "RM" : 8 }})

# Create new features

# 1* Simplifications of existing features. Reducing human subjective grading.

data["SimplOverallQual"] = data.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad

                                                       4 : 2, 5 : 2, 6 : 2, # average

                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good

                                                      })

data["SimplOverallCond"] = data.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad

                                                       4 : 2, 5 : 2, 6 : 2, # average

                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good

                                                      })

data["SimplPoolQC"] = data.PoolQC.replace({1 : 1, 2 : 1, # average

                                             3 : 2, 4 : 2 # good

                                            })

data["SimplGarageCond"] = data.GarageCond.replace({1 : 1, # bad

                                                     2 : 1, 3 : 1, # average

                                                     4 : 2, 5 : 2 # good

                                                    })

data["SimplGarageQual"] = data.GarageQual.replace({1 : 1, # bad

                                                     2 : 1, 3 : 1, # average

                                                     4 : 2, 5 : 2 # good

                                                    })

data["SimplFireplaceQu"] = data.FireplaceQu.replace({1 : 1, # bad

                                                       2 : 1, 3 : 1, # average

                                                       4 : 2, 5 : 2 # good

                                                      })



                                                    

data["SimplFunctional"] = data.Functional.replace({1 : 1, 2 : 1, # bad

                                                     3 : 2, 4 : 2, # major

                                                     5 : 3, 6 : 3, 7 : 3, # minor

                                                     8 : 4 # typical

                                                    })

data["SimplKitchenQual"] = data.KitchenQual.replace({1 : 1, # bad

                                                       2 : 1, 3 : 1, # average

                                                       4 : 2, 5 : 2 # good

                                                      })

data["SimplHeatingQC"] = data.HeatingQC.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

data["SimplBsmtFinType1"] = data.BsmtFinType1.replace({1 : 1, # unfinished

                                                         2 : 1, 3 : 1, # rec room

                                                         4 : 2, 5 : 2, 6 : 2 # living quarters

                                                        })

data["SimplBsmtFinType2"] = data.BsmtFinType2.replace({1 : 1, # unfinished

                                                         2 : 1, 3 : 1, # rec room

                                                         4 : 2, 5 : 2, 6 : 2 # living quarters

                                                        })

data["SimplBsmtCond"] = data.BsmtCond.replace({1 : 1, # bad

                                                 2 : 1, 3 : 1, # average

                                                 4 : 2, 5 : 2 # good

                                                })

data["SimplBsmtQual"] = data.BsmtQual.replace({1 : 1, # bad

                                                 2 : 1, 3 : 1, # average

                                                 4 : 2, 5 : 2 # good

                                                })

data["SimplExterCond"] = data.ExterCond.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

data["SimplExterQual"] = data.ExterQual.replace({1 : 1, # bad

                                                   2 : 1, 3 : 1, # average

                                                   4 : 2, 5 : 2 # good

                                                  })

# 2* Combinations of existing features

# Overall quality of the house

data["OverallGrade"] = data["SimplOverallQual"] * data["SimplOverallCond"] 

# Overall quality of the garage

data["GarageGrade"] = data["SimplGarageQual"]* data["SimplGarageCond"]

# Overall pool score

data["PoolScore"] = data["PoolArea"] * data["PoolQC"]

# Simplified overall quality of the exterior

data["SimplExterGrade"] = data["SimplExterQual"] * data["SimplExterCond"]

# Simplified overall pool score

data["SimplPoolScore"] = data["PoolArea"] * data["SimplPoolQC"]

# Simplified overall garage score

data["SimplGarageScore"] = data["GarageArea"] * data["SimplGarageQual"]

# Simplified overall fireplace score

data["SimplFireplaceScore"] = data["Fireplaces"] * data["SimplFireplaceQu"]

# Simplified overall kitchen score

data["SimplKitchenScore"] = data["KitchenAbvGr"] * data["SimplKitchenQual"]

# Total number of bathrooms

data["TotalBath"] = data["BsmtFullBath"] + data["BsmtHalfBath"] + data["FullBath"] + data["HalfBath"]

# Total SF for house (incl. basement)

data["AllSF"] = data["GrLivArea"] + data["TotalBsmtSF"]

# Total SF for 1st + 2nd floors

data["AllFlrsSF"] = data["1stFlrSF"] + data["2ndFlrSF"]

# Total SF for porch

data["AllPorchSF"] = data["OpenPorchSF"] + data["EnclosedPorch"] + data["3SsnPorch"] + data["ScreenPorch"]

# Has masonry veneer or not

data["HasMasVnr"] = data.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 

                                               "Stone" : 1, "None" : 0})

# House completed before sale or not

data["BoughtOffPlan"] = data.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 

                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
# Differentiate numerical features (minus the target) and categorical features

categorical_features = data.select_dtypes(include = ["object"]).columns

numerical_features = data.select_dtypes(exclude = ["object"]).columns

numerical_features = numerical_features.drop("SalePrice")

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

data_num = data[numerical_features]

data_cat = data[categorical_features]

print(data_num.columns)
# Log transform of the skewed numerical features to lessen impact of outliers

# Inspired by Alexandru Papiu's script : https://www.kaggle.com/apapiu/house-prices-advanced-regression-techniques/regularized-linear-models

# As a general rule of thumb, a skewness with an absolute value > 0.5 is considered at least moderately skewed

skewness = data_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]

print(str(skewness.shape[0]) + " skewed numerical features to log transform")

skewed_features = skewness.index

data_num[skewed_features] = np.log1p(data_num[skewed_features])
#After tranforming the skewed numerical variables, we check for missing data and replace it with madian.

#The mean could have also been used, but since the distribbutions are close to normal it dosen't make to much difference.

print("NAs for numerical features in data : " + str(data_num.isnull().values.sum()))

data_num = data_num.fillna(data_num.median())

print("Remaining NAs for numerical features in data : " + str(data_num.isnull().values.sum()))
# Create dummy features for categorical values via one-hot encoding

print("NAs for categorical features in train : " + str(data_cat.isnull().values.sum()))

data_cat = pd.get_dummies(data_cat)

print("Remaining NAs for categorical features in train : " + str(data_cat.isnull().values.sum()))
# Join categorical and numerical features

data = pd.concat([data_num, data_cat], axis = 1)

print("New number of features : " + str(data.shape[1]))

#creating matrices for sklearn:

train_df = data[:mtrain] #then use it to train

test_df = data[mtrain:] # then use it to make prediction, and output as csv.

# Standardize numerical features

stdSc = StandardScaler()

train_df.loc[:, numerical_features] = stdSc.fit_transform(train_df.loc[:, numerical_features])

#test_.loc[:, numerical_features] = stdSc.transform(X_test.loc[:, numerical_features])

#Creating the cross validation function.



n_folds = 10



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)

    rmse= np.sqrt(-cross_val_score(model, train_df.values, y, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
ridge = Ridge(alpha = 13)

score = rmsle_cv(ridge)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
KRR = KernelRidge(alpha=1, kernel='polynomial', degree=3, coef0=2.5)

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
lasso = Lasso(alpha =0.005, random_state=1, max_iter = 1000)

score = rmsle_cv(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#Using  a very small l1 ratio indicate that the L2 regularization is the dominant.

ENet = ElasticNet(alpha=2, l1_ratio=.000000001, random_state=3)

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
LassoMd = lasso.fit(train_df.values,y)

ENetMd = ENet.fit(train_df.values,y)

ridge_s = ridge.fit(train_df.values, y)

KRRMd = KRR.fit(train_df.values,y)

#test set numerical features standardization and prediction of test set SalePrice.

stdSc = StandardScaler()

test_df.loc[:, numerical_features] = stdSc.fit_transform(test_df.loc[:, numerical_features])

preds = np.expm1(KRRMd.predict(test_df.values))
#create the submission file

test_df = pd.concat([testId, test_df],  axis=1, sort= False).reset_index(drop=True )

sub = pd.DataFrame()

sub['Id'] = testId

sub['SalePrice'] = preds

sub.to_csv('submission.csv',index=False)