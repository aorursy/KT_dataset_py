# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

import matplotlib.style as style

import seaborn as sns

import matplotlib.gridspec as gridspec

import missingno as msno

import seaborn as sns

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor

from scipy.stats import skew  

from scipy import stats

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from IPython.display import display, HTML

%matplotlib inline

from xgboost import XGBRegressor



import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')



# Import Training Data and Test data



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
print ("Size of the train data : {}" .format(train.shape))
print ("Size of the test data : {}" .format(test.shape))
# Information about the features



test.info()
#Save the 'Id' column

train_ID = train['Id']

test_ID = test['Id']



#Now drop the  'Id' colum since it's unnecessary for  the prediction process.

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)


print ("Size of train data after dropping Id: {}" .format(train.shape))

print ("Size of test data after dropping Id: {}" .format(test.shape))

# Statistical about the numerical variables in training data



train.describe().T
# Statistical about the numerical variables in testing data



test.describe().T
# Missing values in training dataset**



def missing_percentage(df):

   

    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])



missing_percentage(train)
# **Missing values in testing dataset**



missing_percentage(test)
#For the PoolQC, 99.52% training data and 99.79% testing data is NA. 

#All the NA in PoolQC is due to no swimming pool.



#"NA" is replaced by "None".  



train["PoolQC"] = train["PoolQC"].fillna('None')

test["PoolQC"] = test["PoolQC"].fillna('None')
# For the MiscFeature, 96.3% training data and 96.5% testing data is NA.  

#As NA referes to no miscellaneous feature, "NA" is replaced by "None".



train["MiscFeature"] = train["MiscFeature"].fillna('None')

test["MiscFeature"] = test["MiscFeature"].fillna('None')
# For the Alley, 93.77% training data and 92.67% testing data is NA.  

#As NA referes to alley access to property , "NA" is replaced by "None".



train["Alley"] = train["Alley"].fillna('None')

test["Alley"] = test["Alley"].fillna('None')
# For the Fence, 80.75% training data and 80.12% testing data is NA.  

#As NA referes to No fence, "NA" is replaced by "None".



train["Fence"] = train["Fence"].fillna('None')

test["Fence"] = test["Fence"].fillna('None')
# For the FireplaceQu, 47.26% training data and 50.03% testing data is NA. 

#As NA referes to No fireplace, "NA" is replaced by "None".



train["FireplaceQu"] = train["FireplaceQu"].fillna('None')

test["FireplaceQu"] = test["FireplaceQu"].fillna('None')
### For the LotFrontage, 17.74% training data and 15.56% testing data is missing. 

#The feature cannot be dropped as the correlation is 0.33 to SalePrice.

# We can fill the missing value with median if outliers exist, otherwise fill it with mean.



# Distribution of training data of Feature LotFrontage 



train.boxplot(column="LotFrontage")
# Distribution of testing data of Feature LotFrontage 



test.boxplot(column="LotFrontage")
#Calculation of mean and median of LotFrontage in training dataset



train['LotFrontage'].mean(),train['LotFrontage'].median()

#Calculation of mean and median of LotFrontage in training dataset



test['LotFrontage'].mean(),test['LotFrontage'].median()
#Fill the missing LotFrontage value with median



train["LotFrontage"] = train["LotFrontage"].fillna(train['LotFrontage'].median())

test["LotFrontage"] = test["LotFrontage"].fillna(test['LotFrontage'].median())
# For the GarageType, GarageCond, GarageFinish, GarageQual 5.55% training data and 5.35% testing data is NA. 

# As NA referes to No garage, "NA" is replaced by "None".



# For the GarageYrBlt, GarageArea and GarageCars, "0" should be filled as there is no garage included



train["GarageType"] = train["GarageType"].fillna('None')

test["GarageType"] = test["GarageType"].fillna('None')

train["GarageFinish"] = train["GarageFinish"].fillna('None')

test["GarageFinish"] = test["GarageFinish"].fillna('None')

train["GarageQual"] = train["GarageQual"].fillna('None')

test["GarageQual"] = test["GarageQual"].fillna('None')

train["GarageCond"] = train["GarageCond"].fillna('None')

test["GarageCond"] = test["GarageCond"].fillna('None')

train["GarageYrBlt"] = train["GarageYrBlt"].fillna('0')

test["GarageYrBlt"] = test["GarageYrBlt"].fillna('0')

train["GarageArea"] = train["GarageArea"].fillna('0')

test["GarageArea"] = test["GarageArea"].fillna('0')

train["GarageCars"] = train["GarageCars"].fillna('0')

test["GarageCars"] = test["GarageCars"].fillna('0')
# For the BsmtFinSF1, BsmtFinSF2, TotalBsmtSF, BsmtFullBath and BsmtHalfBath, The missing values is due to no basement in the house, missing value is filled with "0"



# For BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1 and BsmtFinType2, "none" should be filled as there is no basement



train["BsmtFinSF1"] = train["BsmtFinSF1"].fillna('0')

test["BsmtFinSF1"] = test["BsmtFinSF1"].fillna('0')

train["BsmtFinSF2"] = train["BsmtFinSF2"].fillna('0')

test["BsmtFinSF2"] = test["BsmtFinSF2"].fillna('0')

train["BsmtUnfSF"] = train["BsmtUnfSF"].fillna('0')

test["BsmtUnfSF"] = test["BsmtUnfSF"].fillna('0')

train["TotalBsmtSF"] = train["TotalBsmtSF"].fillna('0')

test["TotalBsmtSF"] = test["TotalBsmtSF"].fillna('0')

train["BsmtFullBath"] = train["BsmtFullBath"].fillna('0')

test["BsmtFullBath"] = test["BsmtFullBath"].fillna('0')

train["BsmtHalfBath"] = train["BsmtHalfBath"].fillna('0')

test["BsmtHalfBath"] = test["BsmtHalfBath"].fillna('0')

train["BsmtQual"] = train["BsmtQual"].fillna('None')

test["BsmtQual"] = test["BsmtQual"].fillna('None')

train["BsmtCond"] = train["BsmtCond"].fillna('None')

test["BsmtCond"] = test["BsmtCond"].fillna('None')

train["BsmtExposure"] = train["BsmtExposure"].fillna('None')

test["BsmtExposure"] = test["BsmtExposure"].fillna('None')

train["BsmtFinType1"] = train["BsmtFinType1"].fillna('None')

test["BsmtFinType1"] = test["BsmtFinType1"].fillna('None')

train["BsmtFinType2"] = train["BsmtFinType2"].fillna('None')

test["BsmtFinType2"] = test["BsmtFinType2"].fillna('None')

# For the MasVnrArea, 0.55% training data and 1.03% testing data is missing value. It should fill by "0" as most likely refer to no masonry veneer.



# For the MasVnrType, 0.55% training data and 1.10% testing data is missing, It should fill by "None" as most likely the house does not include masonry veneer.



train["MasVnrArea"] = train["MasVnrArea"].fillna('0')

test["MasVnrArea"] = test["MasVnrArea"].fillna('0')

train["MasVnrType"] = train["MasVnrType"].fillna('None')

test["MasVnrType"] = test["MasVnrType"].fillna('None')

## For the Electrical, Functional, Utilities, Exterior1st, KitchenQual, Exterior2nd, SaleType Less than 0.2% data is missing. The missing value is filled with the most common value.



train["Electrical"] = train["Electrical"].fillna('sBrKr')

test["Functional"] = test["Functional"].fillna('Typ')

test["Utilities"] = test["Utilities"].fillna('AllPub')

test["Exterior1st"] = test["Exterior1st"].fillna('VinylSd')

test["KitchenQual"] = test["KitchenQual"].fillna('TA')

test["Exterior2nd"] = test["Exterior2nd"].fillna('VinylSd')

test["SaleType"] = test["SaleType"].fillna('Normal')
# For MSZoning, 0.27% testing data with missing value. The value is filled with most common values.



test["MSZoning"] = test["MSZoning"].fillna('RL')
# Double Check missing value in both training and testing data



missing_percentage(train), missing_percentage(test)
train['SalePrice'].describe()
train.hist(column="SalePrice", figsize=(5,5), color="green", bins=100 )
# Distribution of testing data of Target Feature SalePrice 



train.boxplot(column="SalePrice")
#Correct right-skewedness of SalePrice



train.SalePrice = np.log1p(train.SalePrice)
# Plot log SalePrice to check skewedness again



train.hist(column="SalePrice", figsize=(5,5), color="green", bins=100 )
#Find the most correlated features with SalePrice



corrmat = train.corr()

top_correlated_features = corrmat.index[abs(corrmat["SalePrice"])>0.3]

plt.figure(figsize=(20,20))

x = sns.heatmap(train[top_correlated_features].corr(),annot=True,cmap="viridis")
# define plot function, and in this function, we will calculate the skew of X and take the log1p of y

def plot_outlier(x,y):

    tmp=x.dropna()

    skew_value=skew(tmp)

    y=np.log1p(y)

    print('sample lengh: %s   and skew: %s'%(len(x),skew_value))

    fig,axs=plt.subplots(1,2,figsize=(8,3))

    sns.boxplot(x,orient='v',ax=axs[0])

    sns.regplot(x,y,ax=axs[1])

    plt.tight_layout()
# LotFrontage

plot_outlier(train.LotFrontage,train.SalePrice)
# it seems that there are two outlier about LotFrontage, let's remove them and replot 

train=train[train.LotFrontage<300]

plot_outlier(train.LotFrontage,train.SalePrice)
# LotArea

plot_outlier(train.LotArea,train.SalePrice)
# the same thing we do for LotArea, I select threshold from 7000->5000->3000

train=train[train.LotArea<30000]

plot_outlier(train.LotArea,train.SalePrice)
train=train[train.GrLivArea<4500]

plot_outlier(train.GrLivArea,train.SalePrice)
# OverallQual, 

plot_outlier(train.OverallQual,train.SalePrice)
# YearBuilt, newer of the house the higher of the price

plot_outlier(train.YearBuilt,train.SalePrice)
# YearRemodAdd, same as YearBuilt

plot_outlier(train.YearRemodAdd,train.SalePrice)
# BsmtFinSF1, 

plot_outlier(train.BsmtFinSF1,train.SalePrice)
# BsmtUnfSF

plot_outlier(train.BsmtUnfSF,train.SalePrice)
# TotalBsmtSF

plot_outlier(train.TotalBsmtSF,train.SalePrice)
#Outlier at TotalBsmtSF<3000

train=train[train.TotalBsmtSF<3000]

plot_outlier(train.TotalBsmtSF,train.SalePrice)
# finished square feet

BsmtFSF=train.TotalBsmtSF-train.BsmtUnfSF

plot_outlier(BsmtFSF,train.SalePrice)
# 1stFlrSF

plot_outlier(train.loc[:,'1stFlrSF'],train.SalePrice)
# 2ndFlrSF

plot_outlier(train.loc[:,'2ndFlrSF'],train.SalePrice)
# GrLivArea

plot_outlier(train.GrLivArea,train.SalePrice)
#Outlier at GrLivArea < 4000

train=train[train.GrLivArea<4000]

plot_outlier(train.GrLivArea,train.SalePrice)
#BsmtFullBath

plot_outlier(train.BsmtFullBath,train.SalePrice)
#BsmtHalfBath

plot_outlier(train.BsmtHalfBath,train.SalePrice)
#FullBath

plot_outlier(train.FullBath,train.SalePrice)
#HalfBath

plot_outlier(train.HalfBath,train.SalePrice)
#BedroomAbvGr

plot_outlier(train.BedroomAbvGr,train.SalePrice)
#TotRmsAbvGrd

plot_outlier(train.TotRmsAbvGrd,train.SalePrice)
#Fireplaces

plot_outlier(train.Fireplaces,train.SalePrice)
#GarageCars

plot_outlier(train.GarageCars,train.SalePrice)
#GarageArea

plot_outlier(train.GarageArea,train.SalePrice)
#WoodDeckSF

plot_outlier(train.WoodDeckSF,train.SalePrice)
#OpenPorchSF

plot_outlier(train.OpenPorchSF,train.SalePrice)
#EnclosePorch

plot_outlier(train.EnclosedPorch,train.SalePrice)
#3SsnPorch

plot_outlier(train.loc[:,'3SsnPorch'],train.SalePrice)
#ScreenPorch

plot_outlier(train.ScreenPorch,train.SalePrice)
#MoSold

plot_outlier(train.MoSold ,train.SalePrice)
#YrSold

plot_outlier(train.YrSold ,train.SalePrice)
#Deleting outliers in SalePrice



#train=train[(train['GrLivArea'] > 4500) & (train['SalePrice'] < 300000)]

#train=train[(train['1stFlrSF'] > 4000) & (train['SalePrice'] < 300000)]

#train=train[(train['TotalBsmtSF'] >6000) & (train['SalePrice'] < 200000)]
categorial_features = [feature for feature in train.columns if train[feature].dtypes == 'O']

print(categorial_features)


train[categorial_features].head()


for feature in categorial_features:

    print("Feature {} has {} unique values".format(feature, len(train[feature].unique())))

for feature in categorial_features:

    data = train.copy()

    data.groupby(feature)['SalePrice'].median().plot.bar()

    plt.xlabel(feature)

    plt.ylabel('Sale Price')

    plt.title(feature)

    plt.show()

ntrain = train.shape[0]

ntest = test.shape[0]

y_train = train.SalePrice.values

all_data = pd.concat((train, test)).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

print("all_data size is : {}".format(all_data.shape))

all_data['OverallCond'].value_counts()
#Converting some numerical variables that are really categorical type.



#MSSubClass=The building class

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)





#Changing OverallCond into a categorical variable

all_data['OverallCond'] = all_data['OverallCond'].astype(str)





#Year and month sold are transformed into categorical features.

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

        'YrSold', 'MoSold')



for c in cols:

    lbl = LabelEncoder() 

    lbl.fit(list(all_data[c].values)) 

    all_data[c] = lbl.transform(list(all_data[c].values))



# shape        

print('Shape all_data: {}'.format(all_data.shape))
#highly skewed features



numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



# Check the skew of all numerical features

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

print("\nSkew in numerical features: \n")

skewness = pd.DataFrame({'Skew' :skewed_feats})

skewness.head(15)

#Box Cox Transformation of (highly) skewed features



skewness = skewness[abs(skewness) > 0.75]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    #all_data[feat] += 1

    all_data[feat] = boxcox1p(all_data[feat], lam)

all_data = pd.get_dummies(all_data)

all_data.shape

train = all_data[:ntrain]

test = all_data[ntrain:]

train.shape
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

import xgboost as xgb

import lightgbm as lgb
#Validation function

n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)

    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
#KRR = KernelRidge(alpha=0.014, kernel='polynomial', degree=2, coef0=2.5)#

#score = rmsle_cv(KRR)

#print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0004, random_state=1))

#score = rmsle_cv(lasso)

#print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0004, l1_ratio=.9, random_state=3))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
#LassoMd = lasso.fit(train.values,y_train)

ENetMd = ENet.fit(train.values,y_train)

#KRRMd = KRR.fit(train.values,y_train)
finalMd = (np.expm1(ENetMd.predict(test.values))) 
#finalMd = (np.expm1(LassoMd.predict(test.values)) + np.expm1(ENetMd.predict(test.values)) + np.expm1(KRRMd.predict(test.values))) 
sub = pd.DataFrame()

sub['Id'] = test_ID

sub['SalePrice'] = finalMd

sub.to_csv('submission.csv',index=False)