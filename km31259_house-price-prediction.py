# Imports
import pandas as pd
import numpy as np
import scipy
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, make_scorer

from sklearn.linear_model import BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold
import xgboost as xgb
import lightgbm as lgb

from scipy.stats import skew
from scipy.stats import norm
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

#ignore annoying warning (from sklearn and seaborn)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# Definitions
pd.set_option('display.float_format', lambda x: '%.3f' % x)
%matplotlib inline
# Reading the train data
df_train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
df_train.head()
# Shape of train data
df_train.shape
# Reading the train data
df_test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
df_test.head()
# Shape of test data
df_test.shape
# prof = ProfileReport(df_train)
# prof.to_file(output_file = "output.html")
#descriptive statistics summary
df_train['SalePrice'].describe()
#histogram
sns.distplot(df_train['SalePrice']);
#skewness and kurtosis
print("Skewness: %f" % df_train['SalePrice'].skew())
print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#scatter plot grlivarea/saleprice
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#scatter plot totalbsmtsf/saleprice
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
#box plot overallqual/saleprice
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
var = 'YearBuilt'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.xticks(rotation=90);
#correlation matrix
corrmat = df_train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
#saleprice correlation matrix
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#scatterplot
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df_train[cols], size = 2.5)
plt.show();
# Removing SalePrice column from the dataset
df_train2 = df_train.drop(columns=['SalePrice'])

# Adding a flag to Train and Test dataset to separate them later
df_train2['Flag'] = "Train"
df_test2 = df_test
df_test2['Flag'] = "Test"

# Combining the two datasets
df_combo = pd.concat([df_train2,df_test2], ignore_index=True)
print("The shape of the combined dataset:",df_combo.shape)
df_combo.reset_index
df_combo.head()
# missing data in the combined dataset
total = df_combo.isnull().sum().sort_values(ascending=False)
percent = (df_combo.isnull().sum()/df_combo.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data = missing_data[missing_data['Total'] >0]
missing_data
# less than 6 records missing data in the train dataset 
total2 = df_train.isnull().sum().sort_values(ascending=False)
percent2 = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data2 = pd.concat([total2, percent2], axis=1, keys=['Total', 'Percent'])
missing_data2 = missing_data2[(missing_data2['Total'] >0) & (missing_data2['Total'] <=5)]
missing_data2
# Handle missing values for features where median/mean or most common value doesn't make sense

# Dropping the record with NaN in 'Electrical' column in Train dataset
df_combo = df_combo.drop(df_combo[(df_combo['Electrical'].isnull()) & (df_combo['Flag'] == 'Train')].index)
df_train = df_train.dropna(subset = ['Electrical'])
df2 = df_train

# Dropping columns with missing values >= 15% of the # of records
drop_col = missing_data.index[missing_data['Percent']>=0.15]
df_combo = df_combo.drop(columns = drop_col)
                                  
# GarageType etc : data description says NA for garage features is "no garage"
df_combo.loc[:, "GarageCond"] = df_combo.loc[:, "GarageCond"].fillna("No")
df_combo.loc[:, "GarageYrBlt"] = df_combo.loc[:, "GarageYrBlt"].fillna(0)
df_combo.loc[:, "GarageFinish"] = df_combo.loc[:, "GarageFinish"].fillna("No")
df_combo.loc[:, "GarageQual"] = df_combo.loc[:, "GarageQual"].fillna("No")
df_combo.loc[:, "GarageType"] = df_combo.loc[:, "GarageType"].fillna("No")
df_combo.loc[:, "GarageArea"] = df_combo.loc[:, "GarageArea"].fillna(0)
df_combo.loc[:, "GarageCars"] = df_combo.loc[:, "GarageCars"].fillna(0)

# BsmtQual etc : data description says NA for basement features is "no basement"
df_combo.loc[:, "BsmtQual"] = df_combo.loc[:, "BsmtQual"].fillna("No")
df_combo.loc[:, "BsmtCond"] = df_combo.loc[:, "BsmtCond"].fillna("No")
df_combo.loc[:, "BsmtExposure"] = df_combo.loc[:, "BsmtExposure"].fillna("No")
df_combo.loc[:, "BsmtFinType1"] = df_combo.loc[:, "BsmtFinType1"].fillna("No")
df_combo.loc[:, "BsmtFinType2"] = df_combo.loc[:, "BsmtFinType2"].fillna("No")
df_combo.loc[:, "BsmtHalfBath"] = df_combo.loc[:, "BsmtHalfBath"].fillna(0)
df_combo.loc[:, "BsmtFullBath"] = df_combo.loc[:, "BsmtFullBath"].fillna(0)
df_combo.loc[:, "BsmtUnfSF"] = df_combo.loc[:, "BsmtUnfSF"].fillna(0)
df_combo.loc[:, "BsmtFinSF2"] = df_combo.loc[:, "BsmtFinSF2"].fillna(0)
df_combo.loc[:, "TotalBsmtSF"] = df_combo.loc[:, "TotalBsmtSF"].fillna(0)
df_combo.loc[:, "BsmtFinSF1"] = df_combo.loc[:, "BsmtFinSF1"].fillna(0)

# MasVnrType : NA most likely means no veneer
df_combo.loc[:, "MasVnrType"] = df_combo.loc[:, "MasVnrType"].fillna("None")
df_combo.loc[:, "MasVnrArea"] = df_combo.loc[:, "MasVnrArea"].fillna(0)

# Utilities : This column mostly has AllPub as value and won't be helpful
df_combo = df_combo.drop(columns = "Utilities")

# MSZoning (The general zoning classification): 'RL' is by far the most common value. So we can fill in missing values with 'RL'
df_combo.loc[:, "MSZoning"] = df_combo.loc[:, "MSZoning"].fillna(df_combo['MSZoning'].mode()[0])

# Functional
df_combo.loc[:, "Functional"] = df_combo.loc[:, "Functional"].fillna("Typ")

# Exterior1st and Exterior2nd: Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
df_combo.loc[:, "Exterior1st"] = df_combo.loc[:, "Exterior1st"].fillna(df_combo['Exterior1st'].mode()[0])
df_combo.loc[:, "Exterior2nd"] = df_combo.loc[:, "Exterior2nd"].fillna(df_combo['Exterior2nd'].mode()[0])

# KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
df_combo.loc[:, "KitchenQual"] = df_combo.loc[:, "KitchenQual"].fillna(df_combo['KitchenQual'].mode()[0])

# SaleType : Fill in again with most frequent which is "WD"
df_combo['SaleType'] = df_combo['SaleType'].fillna(df_combo['SaleType'].mode()[0])
# Checking if there are anymore NaNs in the dataset
print("Count of NaNs in the combined dataset: ", sum(df_combo.isnull().sum()))
# Keeping a copy of the combined dataset
df_combo_basic = df_combo
# standardizing data
saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
# boxplot for the scaled SalePrice data
plt.figure(figsize=(3, 1))
sp_box = sns.boxplot(x = saleprice_scaled, width = 0.2)
sp_box
# bivariate analysis saleprice/grlivarea
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# deleting points in the train part of the combined dataset
df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]
df_combo = df_combo.drop(df_combo[df_combo['Id'] == 1299].index)
df_combo = df_combo.drop(df_combo[df_combo['Id'] == 524].index)

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)
df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
#bivariate analysis saleprice/grlivarea
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0,800000));
# histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = scipy.stats.probplot(df_train['SalePrice'], plot=plt)
#applying log transformation
df_train['SalePrice'] = np.log(df_train['SalePrice'])
#transformed histogram and normal probability plot
sns.distplot(df_train['SalePrice'], fit=norm);
fig = plt.figure()
res = scipy.stats.probplot(df_train['SalePrice'], plot=plt)
#histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = scipy.stats.probplot(df_train['GrLivArea'], plot=plt)
#data transformation
df_combo['GrLivArea'] = np.log(df_combo['GrLivArea'])
df_train['GrLivArea'] = np.log(df_train['GrLivArea'])
#transformed histogram and normal probability plot
sns.distplot(df_train['GrLivArea'], fit=norm);
fig = plt.figure()
res = scipy.stats.probplot(df_train['GrLivArea'], plot=plt)
#histogram and normal probability plot
sns.distplot(df_train['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = scipy.stats.probplot(df_train['TotalBsmtSF'], plot=plt)
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_train['HasBsmt'] = pd.Series(len(df_train['TotalBsmtSF']), index=df_train.index)
df_train['HasBsmt'] = 0 
df_train.loc[df_train['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_train.loc[df_train['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_train['TotalBsmtSF'])
#create column for new variable (one is enough because it's a binary categorical feature)
#if area>0 it gets 1, for area==0 it gets 0
df_combo['HasBsmt'] = pd.Series(len(df_combo['TotalBsmtSF']), index=df_combo.index)
df_combo['HasBsmt'] = 0 
df_combo.loc[df_combo['TotalBsmtSF']>0,'HasBsmt'] = 1
#transform data
df_combo.loc[df_combo['HasBsmt']==1,'TotalBsmtSF'] = np.log(df_combo['TotalBsmtSF'])
#histogram and normal probability plot
sns.distplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = scipy.stats.probplot(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)
#scatter plot
plt.scatter(df_train['GrLivArea'], df_train['SalePrice']);
#scatter plot
plt.scatter(df_train[df_train['TotalBsmtSF']>0]['TotalBsmtSF'], df_train[df_train['TotalBsmtSF']>0]['SalePrice']);
# Some numerical features are actually really categories
df_combo = df_combo.replace({"MSSubClass" : {20 : "SC20", 30 : "SC30", 40 : "SC40", 45 : "SC45", 
                                       50 : "SC50", 60 : "SC60", 70 : "SC70", 75 : "SC75", 
                                       80 : "SC80", 85 : "SC85", 90 : "SC90", 120 : "SC120", 
                                       150 : "SC150", 160 : "SC160", 180 : "SC180", 190 : "SC190"},
                       "MoSold" : {1 : "Jan", 2 : "Feb", 3 : "Mar", 4 : "Apr", 5 : "May", 6 : "Jun",
                                   7 : "Jul", 8 : "Aug", 9 : "Sep", 10 : "Oct", 11 : "Nov", 12 : "Dec"}
                      })
df_combo = df_combo.replace({
                       "BsmtCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"No" : 0, "Mn" : 1, "Av": 2, "Gd" : 3},
                       "BsmtFinType1" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"No" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, 
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, 
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"No" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
                       "LotShape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       
                       "Street" : {"Grvl" : 1, "Pave" : 2},
                       "Utilities" : {"ELO" : 1, "NoSeWa" : 2, "NoSewr" : 3, "AllPub" : 4}}
                     )
df_combo["SimplOverallQual"] = df_combo.OverallQual.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })

df_combo["SimplOverallCond"] = df_combo.OverallCond.replace({1 : 1, 2 : 1, 3 : 1, # bad
                                                       4 : 2, 5 : 2, 6 : 2, # average
                                                       7 : 3, 8 : 3, 9 : 3, 10 : 3 # good
                                                      })

df_combo["SimplGarageCond"] = df_combo.GarageCond.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })

df_combo["SimplGarageQual"] = df_combo.GarageQual.replace({1 : 1, # bad
                                                     2 : 1, 3 : 1, # average
                                                     4 : 2, 5 : 2 # good
                                                    })

df_combo["SimplFunctional"] = df_combo.Functional.replace({1 : 1, 2 : 1, # bad
                                                     3 : 2, 4 : 2, # major
                                                     5 : 3, 6 : 3, 7 : 3, # minor
                                                     8 : 4 # typical
                                                    })

df_combo["SimplKitchenQual"] = df_combo.KitchenQual.replace({1 : 1, # bad
                                                       2 : 1, 3 : 1, # average
                                                       4 : 2, 5 : 2 # good
                                                      })

df_combo["SimplHeatingQC"] = df_combo.HeatingQC.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })

df_combo["SimplBsmtFinType1"] = df_combo.BsmtFinType1.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })

df_combo["SimplBsmtFinType2"] = df_combo.BsmtFinType2.replace({1 : 1, # unfinished
                                                         2 : 1, 3 : 1, # rec room
                                                         4 : 2, 5 : 2, 6 : 2 # living quarters
                                                        })

df_combo["SimplBsmtCond"] = df_combo.BsmtCond.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })

df_combo["SimplBsmtQual"] = df_combo.BsmtQual.replace({1 : 1, # bad
                                                 2 : 1, 3 : 1, # average
                                                 4 : 2, 5 : 2 # good
                                                })

df_combo["SimplExterCond"] = df_combo.ExterCond.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })

df_combo["SimplExterQual"] = df_combo.ExterQual.replace({1 : 1, # bad
                                                   2 : 1, 3 : 1, # average
                                                   4 : 2, 5 : 2 # good
                                                  })
# Overall quality of the house
df_combo["OverallGrade"] = df_combo["OverallQual"] * df_combo["OverallCond"]

# Overall quality of the garage
df_combo["GarageGrade"] = df_combo["GarageQual"] * df_combo["GarageCond"]

# Overall quality of the exterior
df_combo["ExterGrade"] = df_combo["ExterQual"] * df_combo["ExterCond"]

# Overall kitchen score
df_combo["KitchenScore"] = df_combo["KitchenAbvGr"] * df_combo["KitchenQual"]

# Overall garage score
df_combo["GarageScore"] = df_combo["GarageArea"] * df_combo["GarageQual"]

# Simplified overall quality of the house
df_combo["SimplOverallGrade"] = df_combo["SimplOverallQual"] * df_combo["SimplOverallCond"]

# Simplified overall quality of the exterior
df_combo["SimplExterGrade"] = df_combo["SimplExterQual"] * df_combo["SimplExterCond"]

# Simplified overall garage score
df_combo["SimplGarageScore"] = df_combo["GarageArea"] * df_combo["SimplGarageQual"]

# Simplified overall kitchen score
df_combo["SimplKitchenScore"] = df_combo["KitchenAbvGr"] * df_combo["SimplKitchenQual"]

# Total number of bathrooms
df_combo["TotalBath"] = df_combo["BsmtFullBath"] + (0.5 * df_combo["BsmtHalfBath"]) + \
df_combo["FullBath"] + (0.5 * df_combo["HalfBath"])

# Total SF for house (incl. basement)
df_combo["AllSF"] = df_combo["GrLivArea"] + df_combo["TotalBsmtSF"]

# Total SF for 1st + 2nd floors
df_combo["AllFlrsSF"] = df_combo["1stFlrSF"] + df_combo["2ndFlrSF"]

# Total SF for porch
df_combo["AllPorchSF"] = df_combo["OpenPorchSF"] + df_combo["EnclosedPorch"] + \
df_combo["3SsnPorch"] + df_combo["ScreenPorch"]

# Has masonry veneer or not
df_combo["HasMasVnr"] = df_combo.MasVnrType.replace({"BrkCmn" : 1, "BrkFace" : 1, "CBlock" : 1, 
                                               "Stone" : 1, "None" : 0})

# House completed before sale or not
df_combo["BoughtOffPlan"] = df_combo.SaleCondition.replace({"Abnorml" : 0, "Alloca" : 0, "AdjLand" : 0, 
                                                      "Family" : 0, "Normal" : 0, "Partial" : 1})
# print("Find most important features relative to target")
# corr = df_combo.corr()
# corr.sort_values(["SalePrice"], ascending = False, inplace = True)
# print(corr.SalePrice)
# train["OverallQual-s2"] = train["OverallQual"] ** 2
# train["OverallQual-s3"] = train["OverallQual"] ** 3
# train["OverallQual-Sq"] = np.sqrt(train["OverallQual"])
# train["AllSF-2"] = train["AllSF"] ** 2
# train["AllSF-3"] = train["AllSF"] ** 3
# train["AllSF-Sq"] = np.sqrt(train["AllSF"])
# train["AllFlrsSF-2"] = train["AllFlrsSF"] ** 2
# train["AllFlrsSF-3"] = train["AllFlrsSF"] ** 3
# train["AllFlrsSF-Sq"] = np.sqrt(train["AllFlrsSF"])
# train["GrLivArea-2"] = train["GrLivArea"] ** 2
# train["GrLivArea-3"] = train["GrLivArea"] ** 3
# train["GrLivArea-Sq"] = np.sqrt(train["GrLivArea"])
# train["SimplOverallQual-s2"] = train["SimplOverallQual"] ** 2
# train["SimplOverallQual-s3"] = train["SimplOverallQual"] ** 3
# train["SimplOverallQual-Sq"] = np.sqrt(train["SimplOverallQual"])
# train["ExterQual-2"] = train["ExterQual"] ** 2
# train["ExterQual-3"] = train["ExterQual"] ** 3
# train["ExterQual-Sq"] = np.sqrt(train["ExterQual"])
# train["GarageCars-2"] = train["GarageCars"] ** 2
# train["GarageCars-3"] = train["GarageCars"] ** 3
# train["GarageCars-Sq"] = np.sqrt(train["GarageCars"])
# train["TotalBath-2"] = train["TotalBath"] ** 2
# train["TotalBath-3"] = train["TotalBath"] ** 3
# train["TotalBath-Sq"] = np.sqrt(train["TotalBath"])
# train["KitchenQual-2"] = train["KitchenQual"] ** 2
# train["KitchenQual-3"] = train["KitchenQual"] ** 3
# train["KitchenQual-Sq"] = np.sqrt(train["KitchenQual"])
# train["GarageScore-2"] = train["GarageScore"] ** 2
# train["GarageScore-3"] = train["GarageScore"] ** 3
# train["GarageScore-Sq"] = np.sqrt(train["GarageScore"])
categorical_features = df_combo.select_dtypes(include = ["object"]).columns
numerical_features = df_combo.select_dtypes(exclude = ["object"]).columns

print("Numerical features : " + str(len(numerical_features)))
print("Categorical features : " + str(len(categorical_features)))
df_combo_num = df_combo[numerical_features]
df_combo_cat = df_combo[categorical_features]
# Repeating the same steps for the basic combined dataset
categorical_features_basic = df_combo_basic.select_dtypes(include = ["object"]).columns
numerical_features_basic = df_combo_basic.select_dtypes(exclude = ["object"]).columns

df_combo_basic_num = df_combo_basic[numerical_features_basic]
df_combo_basic_cat = df_combo_basic[categorical_features_basic]
# Handle remaining missing values for numerical features by using median as replacement
print("NAs for numerical features in the combined dataset : " + str(df_combo_num.isnull().values.sum()))
# df_combo_num = df_combo_num.fillna(df_combo_num.median())
print("Remaining NAs for numerical features in the combined dataset : " + str(df_combo_num.isnull().values.sum()))
# skewness = train_num.apply(lambda x: skew(x))
# skewness = skewness[abs(skewness) > 0.5]
# print(str(skewness.shape[0]) + " skewed numerical features to log transform")
# skewed_features = skewness.index
# train_num[skewed_features] = np.log1p(train_num[skewed_features])
print("NAs for categorical features in the combined dataset : " + str(df_combo_cat.isnull().values.sum()))
df_combo_cat2 = df_combo_cat
df_combo_cat2 = df_combo_cat2.drop(columns = "Flag")
df_combo_cat2 = pd.get_dummies(df_combo_cat2)

df_combo_cat = pd.concat([df_combo_cat2, df_combo_cat['Flag']], axis=1)

print("Remaining NAs for categorical features in the combined dataset : " + str(df_combo_cat.isnull().values.sum()))
# Creating dummy variables for the basic combined dataset
df_combo_basic_cat2 = df_combo_basic_cat
df_combo_basic_cat2 = df_combo_basic_cat2.drop(columns = "Flag")
df_combo_basic_cat2 = pd.get_dummies(df_combo_basic_cat2)

df_combo_basic_cat = pd.concat([df_combo_basic_cat2, df_combo_basic_cat['Flag']], axis=1)

print("Remaining NAs for categorical features in the basic combined dataset : " + str(df_combo_basic_cat.isnull().values.sum()))
df_combo2 = pd.concat([df_combo_num, df_combo_cat], axis=1)
df_combo2.shape
# Repeating the same step for basic dataset
df_combo_basic2 = pd.concat([df_combo_basic_num, df_combo_basic_cat], axis=1)
df_combo_basic2.shape
df_train2 = df_combo2[df_combo2['Flag'] == "Train"]
Sp = df_train.loc[:,'SalePrice'].to_frame()

print("Shape 1: ",df_train2.shape)
print("Shape 2: ",Sp.shape)
df_train2.tail(5)
Sp.tail(5)
df_train2 = pd.concat([df_train2, Sp], axis=1)
df_train2.shape
df_train2 = df_train2.set_index('Id')
df_train2.index.names = [None]

df_test2 = df_combo2[df_combo2['Flag'] == "Test"]
df_test2 = df_test2.set_index('Id')
df_test2.index.names = [None]

print("Shape of train dataset: ", df_train2.shape)
print("Shape of test dataset: ", df_test2.shape)
# Repeating the same step for basic dataset
df_train_basic2 = df_combo_basic2[df_combo_basic2['Flag'] == "Train"]

# To avoid using log transformed SalePrice column
df_train_basic2 = pd.concat([df_train_basic2, df2['SalePrice']], axis=1)
df_train_basic2 = df_train_basic2.set_index('Id')
df_train_basic2.index.names = [None]

df_test_basic2 = df_combo_basic2[df_combo_basic2['Flag'] == "Test"]
df_test_basic2 = df_test_basic2.set_index('Id')
df_test_basic2.index.names = [None]

print("Shape of train dataset: ", df_train_basic2.shape)
print("Shape of test dataset: ", df_test_basic2.shape)
cln = list(df_train2.columns)
cln.remove('SalePrice')
cln.remove('Flag')
numerical_features = list(numerical_features)
numerical_features.remove('Id')


cln2 = list(df_train_basic2.columns)
cln2.remove('SalePrice')
cln2.remove('Flag')
numerical_features_basic = list(numerical_features_basic)
numerical_features_basic.remove('Id')
X_train, X_valid, y_train, y_valid = train_test_split(df_train2[cln], df_train2.loc[:,'SalePrice'].to_frame(), test_size=0.25, random_state=13)
X_train2, X_valid2, y_train2, y_valid2 = train_test_split(df_train_basic2[cln2], df_train_basic2.loc[:,'SalePrice'].to_frame(), test_size=0.25, random_state=13)
stdSc = StandardScaler()

X_train.loc[:, numerical_features] = stdSc.fit_transform(X_train.loc[:, numerical_features])
X_valid.loc[:, numerical_features] = stdSc.transform(X_valid.loc[:, numerical_features])
df_test2.loc[:, numerical_features] = stdSc.transform(df_test2.loc[:, numerical_features])

# y_train = stdSc.fit_transform(y_train)
# y_valid = stdSc.transform(y_valid)
stdSc_basic = StandardScaler()

X_train2.loc[:, numerical_features_basic] = stdSc_basic.fit_transform(X_train2.loc[:, numerical_features_basic])
X_valid2.loc[:, numerical_features_basic] = stdSc_basic.transform(X_valid2.loc[:, numerical_features_basic])
df_test_basic2.loc[:, numerical_features_basic] = stdSc_basic.transform(df_test_basic2.loc[:, numerical_features_basic])

y_train2 = pd.Series((np.ravel(stdSc_basic.fit_transform(y_train2)) + 12), index= X_train2.index)
y_valid2 = pd.Series((np.ravel(stdSc_basic.transform(y_valid2)) + 12), index= X_valid2.index)
# Checking if all X_train looks good

cf = X_train.select_dtypes(include = ["object"]).columns
nf = X_train.select_dtypes(exclude = ["object"]).columns

print("Numerical features : " + str(len(nf)))
print("Categorical features : " + str(len(cf)))
y_train2.describe()
# boxplot for the train SalePrice data
plt.figure(figsize=(3, 1))
yt_box = sns.boxplot(x = y_train2, width = 0.2)
yt_box
y_valid2.describe()
# boxplot for the validation SalePrice data
plt.figure(figsize=(3, 1))
yt_box = sns.boxplot(x = y_valid2, width = 0.2)
yt_box
# Define error measure for official scoring : RMSE
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv_train(model, X_train, y_train, cv=5):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = scorer, cv = 5))
    return(rmse)

# def rmse_cv_valid(model, X_valid, y_valid, cv=5):
#     rmse= np.sqrt(-cross_val_score(model, X_valid, y_valid, scoring = scorer, cv = 5))
#     return(rmse)
# We use a validation dataset to validate which model/ model parameter is the best one.

def Linear_regression_no_reg(X_train, y_train, X_valid, y_valid):
    # Linear Regression
    lr = LinearRegression()
    lr2 = LinearRegression()
    lr.fit(X_train, y_train)
    
    rmse_train = rmse_cv_train(lr2, X_train, y_train).mean()
    rmse_valid = np.sqrt(mean_squared_error(y_valid, lr.predict(X_valid)))
    
    r2_train = cross_val_score(lr2, X_train, y_train, cv=5).mean()
    r2_valid = lr.score(X_valid, y_valid)

    # Results from the fit model
    print("RMSE on Training set without error cross validation :", np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))
    print("RMSE on Validation set without error cross validation :", rmse_valid)
    print("")
    print("RMSE on Training set with error cross validation :", rmse_train)
#     print("RMSE on Validation set with error cross validation :", rmse_valid)
    print("")
    print("R2 on Training set without error cross validation :", lr.score(X_train, y_train))
    print("R2 on Validation set without error cross validation :", r2_valid)
    print("")
    print("R2 on Training set with error cross validation :", r2_train)
#     print("R2 on Validation set with error cross validation :", r2_valid)
    
    y_train_pred = np.ravel(lr.predict(X_train))
    y_valid_pred = np.ravel(lr.predict(X_valid))


#     # Plot residuals
#     plt.scatter(y_train_pred, y_train_pred - np.ravel(y_train.to_numpy()), c = "blue", marker = "s", label = "Training data")
#     plt.scatter(y_valid_pred, y_valid_pred - np.ravel(y_valid.to_numpy()), c = "lightgreen", marker = "s", label = "Validation data")
#     plt.title("Linear regression")
#     plt.xlabel("Predicted values")
#     plt.ylabel("Residuals")
#     plt.legend(loc = "upper left")
#     plt.hlines(y = 0, xmin = 9, xmax = 15, color = "red")
#     plt.show()

#     # Plot predictions
#     plt.scatter(y_train_pred, np.ravel(y_train.to_numpy()), c = "blue", marker = "s", label = "Training data")
#     plt.scatter(y_valid_pred, np.ravel(y_valid.to_numpy()), c = "lightgreen", marker = "s", label = "Validation data")
#     plt.title("Linear regression")
#     plt.xlabel("Predicted values")
#     plt.ylabel("Real values")
#     plt.legend(loc = "upper left")
#     plt.plot([9, 15], [9, 15], c = "red")
#     plt.show()
    return(rmse_train, rmse_valid, r2_train, r2_valid, y_valid_pred)
LR_no_reg = Linear_regression_no_reg(X_train, y_train, X_valid, y_valid)
def Linear_regression_Ridge(X_train, y_train, X_valid, y_valid):    
    
    # Ridge Linear Regression
    ridge = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    print("Best alpha :", alpha)

    print("Try again for more precision with alphas centered around " + str(alpha))
    ridge = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, alpha * .85, 
                          alpha * .9, alpha * .95, alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15,
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                    cv = 5)
    ridge.fit(X_train, y_train)
    alpha = ridge.alpha_
    coef = np.ravel(ridge.coef_)
    print("Best alpha :", alpha)
    print("")
    
    ridge2 = Ridge(alpha = alpha)
    
    rmse_train = rmse_cv_train(ridge2, X_train, y_train).mean()
    rmse_valid = np.sqrt(mean_squared_error(y_valid, ridge.predict(X_valid)))
    
    r2_train = cross_val_score(ridge2, X_train, y_train, cv=5).mean()
    r2_valid = ridge.score(X_valid, y_valid)
    
    coefs = pd.Series(coef, index = X_train.columns)
  
    # Results from the fit model
    print("RMSE on Training set without error cross validation :", np.sqrt(mean_squared_error(y_train, ridge.predict(X_train))))
    print("RMSE on Validation set without error cross validation :", rmse_valid)
    print("")
    print("RMSE on Training set with error cross validation :", rmse_train)
#     print("RMSE on Validation set with error cross validation :", rmse_valid)
    print("")
    print("R2 on Training set without error cross validation :", ridge.score(X_train, y_train))
    print("R2 on Validation set without error cross validation :", r2_valid)
    print("")
    print("R2 on Training set with error cross validation :", r2_train)
#     print("R2 on Validation set with error cross validation :", r2_valid)
 
    y_train_rdg = np.ravel(ridge.predict(X_train))
    y_valid_rdg = np.ravel(ridge.predict(X_valid))

    # Plot residuals
    plt.scatter(y_train_rdg, y_train_rdg - np.ravel(y_train.to_numpy()), c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_valid_rdg, y_valid_rdg - np.ravel(y_valid.to_numpy()), c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc = "upper left")
    plt.hlines(y = 0, xmin = 9, xmax = 15, color = "red")
    plt.show()

    # Plot predictions
    plt.scatter(y_train_rdg, np.ravel(y_train.to_numpy()), c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_valid_rdg, np.ravel(y_valid.to_numpy()), c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression with Ridge regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.plot([9, 15], [9, 15], c = "red")
    plt.show()
    
    # Plot important coefficients
    print("")
    print("Ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                         coefs.sort_values().tail(10)])
    imp_coefs.plot(kind = "barh")
    plt.title("Coefficients in the Ridge Model")
    plt.show()
    return(rmse_train, rmse_valid, r2_train, r2_valid, y_valid_rdg)
LR_Ridge = Linear_regression_Ridge(X_train, y_train, X_valid, y_valid)
def Linear_regression_Lasso(X_train, y_train, X_valid, y_valid):    
    
    # Lasso Linear Regression
    lasso = LassoCV(alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 
                          0.3, 0.6, 1], 
                max_iter = 50000, cv = 5)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    print("Best alpha :", alpha)

    print("Try again for more precision with alphas centered around " + str(alpha))
    lasso = LassoCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, alpha * .8, 
                          alpha * .85, alpha * .9, alpha * .95, alpha, alpha * 1.05, 
                          alpha * 1.1, alpha * 1.15, alpha * 1.25, alpha * 1.3, alpha * 1.35, 
                          alpha * 1.4], 
                max_iter = 50000, cv = 5)
    lasso.fit(X_train, y_train)
    alpha = lasso.alpha_
    coef = np.ravel(lasso.coef_)
    print("Best alpha :", alpha)
    print("")
    
    lasso2 = Lasso(alpha = alpha)
    
    rmse_train = rmse_cv_train(lasso2, X_train, y_train).mean()
    rmse_valid = np.sqrt(mean_squared_error(y_valid, lasso.predict(X_valid)))
    
    r2_train = cross_val_score(lasso2, X_train, y_train, cv=5).mean()
    r2_valid = lasso.score(X_valid, y_valid)
    
    coefs = pd.Series(coef, index = X_train.columns)
  
    # Results from the fit model
    print("RMSE on Training set without error cross validation :", np.sqrt(mean_squared_error(y_train, lasso.predict(X_train))))
    print("RMSE on Validation set without error cross validation :", rmse_valid)
    print("")
    print("RMSE on Training set with error cross validation :", rmse_train)
#     print("RMSE on Validation set with error cross validation :", rmse_valid)
    print("")
    print("R2 on Training set without error cross validation :", lasso.score(X_train, y_train))
    print("R2 on Validation set without error cross validation :", r2_valid)
    print("")
    print("R2 on Training set with error cross validation :", r2_train)
#     print("R2 on Validation set with error cross validation :", r2_valid)

    y_train_las = np.ravel(lasso.predict(X_train))
    y_valid_las = np.ravel(lasso.predict(X_valid))

    # Plot residuals
    plt.scatter(y_train_las, y_train_las - np.ravel(y_train.to_numpy()), c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_valid_las, y_valid_las - np.ravel(y_valid.to_numpy()), c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc = "upper left")
    plt.hlines(y = 0, xmin = 9, xmax = 15, color = "red")
    plt.show()

    # Plot predictions
    plt.scatter(y_train_las, np.ravel(y_train.to_numpy()), c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_valid_las, np.ravel(y_valid.to_numpy()), c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression with Lasso regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.plot([9, 15], [9, 15], c = "red")
    plt.show()
    
    # Plot important coefficients
    print("")
    print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  \
          str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                         coefs.sort_values().tail(10)])
    imp_coefs.plot(kind = "barh")
    plt.title("Coefficients in the Lasso Model")
    plt.show()
    return(rmse_train, rmse_valid, r2_train, r2_valid, y_valid_las)
LR_Lasso = Linear_regression_Lasso(X_train, y_train, X_valid, y_valid)
def Linear_regression_ElasticNet(X_train, y_train, X_valid, y_valid):
    # ElasticNet
    elasticNet = ElasticNetCV(l1_ratio = [0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
                              alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 
                                        0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                              max_iter = 50000, cv = 5)
    elasticNet.fit(X_train, y_train)
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )

    print("Try again for more precision with l1_ratio centered around " + str(ratio))
    elasticNet = ElasticNetCV(l1_ratio = [ratio * .85, ratio * .9, ratio * .95, ratio, ratio * 1.05, ratio * 1.1, ratio * 1.15],
                              alphas = [0.0001, 0.0003, 0.0006, 0.001, 0.003, 0.006, 0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6], 
                              max_iter = 50000, cv = 5)
    elasticNet.fit(X_train, y_train)
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
                              max_iter = 50000, cv = 5)
    elasticNet.fit(X_train, y_train)
    if (elasticNet.l1_ratio_ > 1):
        elasticNet.l1_ratio_ = 1    
    alpha = elasticNet.alpha_
    ratio = elasticNet.l1_ratio_
    print("Best l1_ratio :", ratio)
    print("Best alpha :", alpha )
    print("")
    
    elasticNet2 = ElasticNet(l1_ratio = ratio, alpha = alpha)
    
    rmse_train = rmse_cv_train(elasticNet2, X_train, y_train).mean()
    rmse_valid = np.sqrt(mean_squared_error(y_valid, elasticNet.predict(X_valid)))
    
    r2_train = cross_val_score(elasticNet2, X_train, y_train, cv=5).mean()
    r2_valid = elasticNet.score(X_valid, y_valid)
  
    # Results from the fit model
    print("RMSE on Training set without error cross validation :", np.sqrt(mean_squared_error(y_train, elasticNet.predict(X_train))))
    print("RMSE on Validation set without error cross validation :", rmse_valid)
    print("")
    print("RMSE on Training set with error cross validation :", rmse_train)
#     print("RMSE on Validation set with error cross validation :", rmse_valid)
    print("")
    print("R2 on Training set without error cross validation :", elasticNet.score(X_train, y_train))
    print("R2 on Validation set without error cross validation :", r2_valid)
    print("")
    print("R2 on Training set with error cross validation :", r2_train)
#     print("R2 on Validation set with error cross validation :", r2_valid)
    
    y_train_ela = np.ravel(elasticNet.predict(X_train))
    y_valid_ela = np.ravel(elasticNet.predict(X_valid))

    # Plot residuals
    plt.scatter(y_train_ela, y_train_ela - np.ravel(y_train.to_numpy()), c = "blue", marker = "s", label = "Training data")
    plt.scatter(y_valid_ela, y_valid_ela - np.ravel(y_valid.to_numpy()), c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.legend(loc = "upper left")
    plt.hlines(y = 0, xmin = 9, xmax = 15, color = "red")
    plt.show()

    # Plot predictions
    plt.scatter(np.ravel(y_train.to_numpy()), y_train_ela, c = "blue", marker = "s", label = "Training data")
    plt.scatter(np.ravel(y_valid.to_numpy()), y_valid_ela, c = "lightgreen", marker = "s", label = "Validation data")
    plt.title("Linear regression with ElasticNet regularization")
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc = "upper left")
    plt.plot([9, 15], [9, 15], c = "red")
    plt.show()

    # Plot important coefficients
    coefs = pd.Series(np.ravel(elasticNet.coef_), index = X_train.columns)
    print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +  str(sum(coefs == 0)) + " features")
    imp_coefs = pd.concat([coefs.sort_values().head(10),
                         coefs.sort_values().tail(10)])
    imp_coefs.plot(kind = "barh")
    plt.title("Coefficients in the ElasticNet Model")
    plt.show()
    
    return(rmse_train, rmse_valid, r2_train, r2_valid, y_valid_ela)
LR_ElasticNet = Linear_regression_ElasticNet(X_train, y_train, X_valid, y_valid)
avg_valid_pred = np.mean([LR_Lasso[4],LR_ElasticNet[4]], axis = 0)
avg_valid_pred.shape
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))
rmsle(y_valid, avg_valid_pred)
# # Ridge
# Basic_LR_Ridge = Linear_regression_Ridge(X_train2, y_train2, X_valid2, y_valid2)
# # Lasso
# Basic_LR_Lasso = Linear_regression_Lasso(X_train2, y_train2, X_valid2, y_valid2)
# # ElasticNet
# Basic_LR_ElasticNet = Linear_regression_ElasticNet(X_train2, y_train2, X_valid2, y_valid2)
res = [LR_no_reg[0:4], LR_Ridge[0:4], LR_Lasso[0:4], LR_ElasticNet[0:4]] #, Basic_LR_Ridge, Basic_LR_Lasso, Basic_LR_ElasticNet]
res2 = pd.DataFrame(res, columns =['rmse_train', 'rmse_valid', 'r2_train', 'r2_valid'])
res2['Data_Model'] = list(['LR_no_reg', 'LR_Ridge', 'LR_Lasso', 'LR_ElasticNet']) #, 'Basic_LR_Ridge', 'Basic_LR_Lasso', 'Basic_LR_ElasticNet'])
cols = ['Data_Model', 'rmse_train', 'r2_train', 'rmse_valid','r2_valid']
res2 = res2[cols]
res2
res2 = res2.drop(index = 0)
res2 = res2.sort_values(by=['rmse_valid'], ascending=True)
res2
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=13).get_n_splits(X_train.values)
    rmse= np.sqrt(-cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
lasso = Lasso(alpha =0.0008)
ENet = ElasticNet(alpha= 0.01, l1_ratio= 0.0475)
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
score = rmsle_cv(lasso)
print("\nLasso training score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(ENet)
print("ElasticNet training score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(KRR)
print("Kernel Ridge training score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(GBoost)
print("Gradient Boosting training score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_xgb)
print("Xgboost training score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM training score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
score = rmsle_cv(averaged_models)
print(" Averaged base models training score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))
averaged_models.fit(X_train, y_train)
averaged_valid_pred = averaged_models.predict(X_valid)
print(" Averaged base models validation score: {:.4f}\n".format(np.sqrt(mean_squared_error(y_valid, averaged_valid_pred))))
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)
# stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),
#                                                  meta_model = lasso)

# score = rmsle_cv(stacked_averaged_models)
# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# def rmsle(y, y_pred):
#     return np.sqrt(mean_squared_error(y, y_pred))
# stacked_averaged_models.fit(X_train, y_train)
# stacked_train_pred = stacked_averaged_models.predict(X_train.values)
# stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))
# print(rmsle(y_train, stacked_train_pred))
# model_xgb.fit(X_train, y_train)
# xgb_train_pred = model_xgb.predict(X_train)
# xgb_pred = np.expm1(model_xgb.predict(df_test2.values))
# print(rmsle(y_train, xgb_train_pred))
# model_lgb.fit(X_train, y_train)
# lgb_train_pred = model_lgb.predict(train)
# lgb_pred = np.expm1(model_lgb.predict(df_test2.values))
# print(rmsle(y_train, lgb_train_pred))
# '''RMSE on the entire Train data when averaging'''

# print('RMSLE score on train data:')
# print(rmsle(y_train,stacked_train_pred*0.70 +
#                xgb_train_pred*0.15 + lgb_train_pred*0.15 ))
# ensemble = stacked_pred*0.70 + xgb_pred*0.15 + lgb_pred*0.15
