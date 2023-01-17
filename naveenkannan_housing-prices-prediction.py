#loading all the required packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
#loading the train and test dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train_raw = train.copy()

test_raw = test.copy()

print("Train : {}".format(train.shape))

print("Test : {}".format(test.shape))
train.head()
test.head()
#some useful information about the data

train.info()
test.info()
#outliers in grlivarea (indicated in dataset documentation)

sns.scatterplot(x = train.GrLivArea,y = train.SalePrice)
#taking out outliers

train.drop(index = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index,

           inplace = True)

sns.scatterplot(x = train.GrLivArea,y = train.SalePrice)
train.SalePrice.describe()
#histogram plot

from scipy.stats import norm

sns.distplot(train.SalePrice,fit = norm)

print('Skewness : %f' %train.SalePrice.skew())
#log transforming saleprice

train['SalePrice'] = np.log1p(train.SalePrice)

sns.distplot(train.SalePrice,fit = norm)

print('Skewness : %f' %train.SalePrice.skew())
#missing values in training set

train.isnull().sum().sort_values(ascending = False)[:20]
#missing values in test set

test.isnull().sum().sort_values(ascending = False)[:35]
n_train = train.shape[0]

n_test = test.shape[0]

y_train = train.SalePrice.values

train.drop(columns = 'SalePrice',inplace = True)

all_data = pd.concat((train,test)).reset_index(drop = True)

print("all_data size is : {}".format(all_data.shape))
all_data.MSZoning.fillna(all_data.MSZoning.mode()[0],inplace = True)

#all_data.MSZoning.isnull().sum()
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x : 

                                                                                   x.fillna(x.median()))
all_data.Alley.fillna('None',inplace = True)
all_data.PoolQC.fillna('None',inplace = True)
all_data.MiscFeature.fillna('None',inplace = True)
all_data.Fence.fillna('None',inplace = True)
all_data.FireplaceQu.fillna('None',inplace = True)
for feature in ('GarageType','GarageFinish','GarageQual','GarageCond') :

    all_data[feature].fillna('None',inplace = True)

for feature in ('GarageYrBlt','GarageArea','GarageCars') :

    all_data[feature].fillna(0,inplace = True)    
for feature in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[feature].fillna(0,inplace = True)

for feature in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[feature].fillna('None',inplace = True)
all_data["MasVnrType"].fillna("None",inplace = True)

all_data["MasVnrArea"].fillna(0,inplace = True)
all_data['Electrical'].fillna(all_data['Electrical'].mode()[0],inplace = True)
all_data['Utilities'].fillna(all_data['Utilities'].mode()[0],inplace = True)
all_data['Functional'].fillna('Typ',inplace = True)
all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0],inplace = True)
all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0],inplace = True)

all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0],inplace = True)
all_data['SaleType'].fillna(all_data['SaleType'].mode()[0],inplace = True)
#some numerical features are actually categorical

for feature in ('MSSubClass','MoSold') :

    all_data[feature] = all_data[feature].astype(str)
#mapping the quality variables to numerical

#lower quality to higher quality mapped to lower number to higher number

qual_dict = {'None': 0,'Po': 1,'Fa': 2,'TA': 3,'Gd': 4,'Ex': 5 }

for feat in ('ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC','KitchenQual','FireplaceQu',

            'GarageQual','GarageCond','PoolQC') :

    all_data[feat] = all_data[feat].map(qual_dict).astype(int)

    

#further mapping

all_data["BsmtExposure"] = all_data["BsmtExposure"].map(

    {"None": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}).astype(int)

all_data["Street"] = all_data["Street"].map({'None': 0 , 'Grvl': 1 , 'Pave': 2}).astype(int)

all_data["Alley"] = all_data["Alley"].map({'None': 0 , 'Grvl': 1 , 'Pave': 2}).astype(int)



bsmtfin_dict = {"None": 0, "Unf": 1, "LwQ": 2, "Rec": 3, "BLQ": 4, "ALQ": 5, "GLQ": 6}

all_data["BsmtFinType1"] = all_data["BsmtFinType1"].map(bsmtfin_dict).astype(int)

all_data["BsmtFinType2"] = all_data["BsmtFinType2"].map(bsmtfin_dict).astype(int)



all_data["CentralAir"] = all_data["CentralAir"].map({"N": 0 , "Y": 1})

all_data["Functional"] = all_data["Functional"].map({"Sal": 1, "Sev": 1, "Maj2": 2, "Maj1": 2, 

         "Mod": 3, "Min2": 3, "Min1": 3, "Typ": 4}).astype(int)

all_data["GarageFinish"] = all_data["GarageFinish"].map(

    {"None": 0, "Unf": 1, "RFn": 2,"Fin": 3}).astype(int)

all_data["Fence"] = all_data["Fence"].map(

        {"None": 0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv": 4}).astype(int)
#Extracting Features

#LotShape - Distinguishing just between regular and irregular as IR3 and IR2 appears rarely

all_data['IsRegularLotShape'] = (all_data['LotShape'] == 'Reg') * 1

#LandContour - Most properties are level

all_data['IsLandLevel'] = (all_data['LandContour'] == 'Lvl') * 1

#LandSlope - most land slopes are gentle

all_data['IsGentleSlope'] = (all_data['LandSlope'] == 'Gtl') * 1

#Electrical - Most properties have standard circuit breakers

all_data['IsElectricalSBrkr'] = (all_data['Electrical'] == 'SBrkr') * 1

#GarageType - Almost all the properties have attached garage

all_data['IsGargageDetached'] = (all_data['GarageType'] == 'Detchd') * 1

#MiscFeature - Presence or absence

all_data['HasMiscFeature'] = (all_data['MiscFeature'] != 'None') * 1

#PavedDrive - Ispaved drive or not

all_data['IsPavedDrive'] = (all_data['PavedDrive'] == 'Y') * 1

#House not completed

all_data['HouseNotCompleted'] = (all_data['SaleCondition'] == 'Partial') * 1



# complete information is extracted from these two , so dropping these features

dropcols = ['MiscFeature','PavedDrive']

all_data.drop(dropcols,axis = 1,inplace = True)
#Creating new features

#whether the house was remodelled once

all_data['Remodelled'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']) * 1

#whether it is a new house

all_data['NewHouse'] = (all_data['YearBuilt'] == all_data['YrSold']) * 1

#whether remodelled recently

all_data['RecentRemodel'] = (all_data['YearRemodAdd'] == all_data['YrSold']) * 1

#Total sqft for house

all_data['AllSF'] = all_data['GrLivArea'] + all_data['TotalBsmtSF']

#Total full bathrooms

all_data['TotalFullBath'] = all_data['FullBath'] + all_data['BsmtFullBath']

#Total half bathrooms

all_data['TotalHalfBath'] = all_data['HalfBath'] + all_data['BsmtHalfBath']

#All floors sqft

all_data['AllFlrSF'] = all_data['1stFlrSF'] + all_data['2ndFlrSF']



#Handling year variables

for feat in ('YearBuilt','YearRemodAdd','GarageYrBlt','YrSold') :

    all_data[feat] = pd.qcut(all_data[feat],q = 10,duplicates = 'drop')
#splitting up into train and test sets

train = all_data[:n_train]

test = all_data[n_train:]
#applying log transformation for highly skewed features

from scipy.stats import skew

numeric_features = train.select_dtypes(exclude = [object,'category']).drop(columns = ['Id']).columns

skewness = train[numeric_features].apply(lambda x : skew(x.dropna()))

skewed_features = skewness[abs(skewness) > 0.75].index

train[skewed_features] = np.log1p(train[skewed_features])

test[skewed_features] = np.log1p(test[skewed_features])
train = pd.get_dummies(train)

test = pd.get_dummies(test)
print(train.shape)

print(test.shape)
common_features = train.columns & test.columns

train = train[common_features]

test = test[common_features]
#defining our error metric

from sklearn.model_selection import KFold,cross_val_score

def rmse(model) :

    kf = KFold(n_splits = 5,shuffle = True,random_state = 6)

    rmse = np.sqrt(-1 * cross_val_score(model,X = train,y = y_train,

                                        scoring = "neg_mean_squared_error",cv = kf))

    return rmse

    
#Linear Regression

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

print("Linear Regression score : {:.4f}".format(rmse(lr).mean()))

lr.fit(train,y_train)

pred_lr = np.exp(lr.predict(test))

submission_lr = pd.DataFrame({"Id" : test_raw["Id"], "SalePrice" : pred_lr })

submission_lr.to_csv("linear_regression",index = False)
#Let's do cross validation to find the best alpha

# from sklearn.linear_model import Ridge, RidgeCV

# kfold = KFold(n_splits = 5,shuffle = True,random_state = 6)

# ridgecv = RidgeCV(alphas = [0.01,0.03,0.06,0.1,0.3,0.6,1,3,6,10,30,60],

#                   scoring = 'neg_mean_squared_error',cv = kfold)

# ridgecv.fit(train,y_train)

# print("Best alpha {}".format(ridgecv.alpha_))

# #further cross validation with alphas centred around best alpha

# alphas = [coeff * ridgecv.alpha_ for coeff in np.arange(0.6,1.4,0.05)]

# ridgecv = RidgeCV(alphas = alphas,scoring = 'neg_mean_squared_error',cv = kfold)

# ridgecv.fit(train,y_train)

# print("Best alpha after further cross validation {}".format(ridgecv.alpha_))
#fitting ridge regression

from sklearn.linear_model import Ridge, RidgeCV

ridge = Ridge(alpha = 6.3) #found by cross validation

print("Ridge Regression score : {:.4f}".format(rmse(ridge).mean()))

ridge.fit(train,y_train)

pred_rr = np.exp(ridge.predict(test))

submission_rr = pd.DataFrame({"Id" : test_raw["Id"], "SalePrice" : pred_rr })

submission_rr.to_csv("ridge_regression",index = False)
#cross validation to find optimum alpha for lasso regression

# from sklearn.linear_model import LassoCV,Lasso

# kfold = KFold(n_splits = 5,shuffle = True,random_state = 6)

# lassocv = LassoCV(alphas = [0.0001,0.0003,0.0006,0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3,0.6,1,3,6],

#                  max_iter = 50000,cv = kfold)

# lassocv.fit(train,y_train)

# print("Best alpha {}".format(lassocv.alpha_))

# #further cross validation with alphas centred around best alpha

# alphas = [coeff * lassocv.alpha_ for coeff in np.arange(0.6,1.4,0.05)]

# lassocv = LassoCV(alphas = alphas,max_iter = 50000,cv = kfold)

# lassocv.fit(train,y_train)

# print("Best alpha after further cross validation {}".format(lassocv.alpha_))
#fitting lasso regression

from sklearn.linear_model import LassoCV,Lasso

lasso = Lasso(alpha = 0.00039,max_iter = 50000) #found by cross validation

print("Lasso Regression score : {:.4f}".format(rmse(lasso).mean()))

lasso.fit(train,y_train)

pred_lasso = np.exp(lasso.predict(test))

submission_lasso = pd.DataFrame({"Id" : test_raw["Id"], "SalePrice" : pred_lasso })

submission_lasso.to_csv("lasso_regression",index = False)
#cross validation to find the optimum alpha and l1_ratio

# from sklearn.linear_model import ElasticNetCV, ElasticNet

# kfold = KFold(n_splits = 5,shuffle = True,random_state = 6)

# encv = ElasticNetCV(alphas = [0.0001,0.0003,0.0006,0.001,0.003,0.006,0.01,0.03,0.06,0.1,0.3,0.6,1,3,6],

#                    l1_ratio = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1],

#                     max_iter = 50000,cv = kfold)

# encv.fit(train,y_train)

# print("Best alpha {}".format(encv.alpha_))

# print("Best l1_ratio {}".format(encv.l1_ratio_))

# #further cross validation

# alphas = [coeff * encv.alpha_ for coeff in np.arange(0.6,1.4,0.05)]

# l1_ratios = [coeff * encv.l1_ratio_ for coeff in np.arange(0.6,1.4,0.05)]

# encv = ElasticNetCV(alphas = alphas,l1_ratio = l1_ratios,max_iter = 50000,cv = kfold)

# encv.fit(train,y_train)

# print("Best alpha after further cross validation {}".format(encv.alpha_))

# print("Best l1_ratio after further cross validation {}".format(encv.l1_ratio_))

# #further narrowing

# alphas = [coeff * encv.alpha_ for coeff in np.arange(0.6,1.4,0.05)]

# l1_ratios = [coeff * encv.l1_ratio_ for coeff in np.arange(0.9,1.2,0.05)]

# encv = ElasticNetCV(alphas = alphas,l1_ratio = l1_ratios,max_iter = 50000,cv = kfold)

# encv.fit(train,y_train)

# print("Best alpha after further cross validation {}".format(encv.alpha_))

# print("Best l1_ratio after further cross validation {}".format(encv.l1_ratio_))

# #further narrowing (as it is best to keep l1_ratio near to 1 (that is more inclined towards Lasso))

# alphas = [coeff * encv.alpha_ for coeff in np.arange(0.6,1.4,0.05)]

# l1_ratios = [coeff * encv.l1_ratio_ for coeff in np.arange(0.9,1.1,0.05)]

# encv = ElasticNetCV(alphas = alphas,l1_ratio = l1_ratios,max_iter = 50000,cv = kfold)

# encv.fit(train,y_train)

# print("Best alpha after further cross validation {}".format(encv.alpha_))

# print("Best l1_ratio after further cross validation {}".format(encv.l1_ratio_))

#Both l1_ratio and alpha are converged
#fitting ElasticNet Regression

from sklearn.linear_model import ElasticNetCV, ElasticNet

en = ElasticNet(alpha = 0.000432,l1_ratio = 0.891,

               max_iter = 50000) #found by a series of cross validation and narrowing down

print("ElasticNet Regression score : {:.4f}".format(rmse(en).mean()))

en.fit(train,y_train)

pred_en = np.exp(en.predict(test))

submission_en = pd.DataFrame({"Id" : test_raw["Id"], "SalePrice" : pred_en})

submission_en.to_csv("ElasticNet_regression",index = False)