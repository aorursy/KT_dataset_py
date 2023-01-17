# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from datetime import datetime

from scipy.stats import skew

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV, Lasso, Ridge, ElasticNet

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler, StandardScaler

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split

from sklearn.metrics import mean_squared_error

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

import matplotlib.pyplot as plt

import scipy.stats as stats

import sklearn.linear_model as linear_model

from sklearn.manifold import TSNE

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

import matplotlib.gridspec as gridspec

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

%matplotlib inline
# Importing train data



train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

train.head(6) # Mention no of rows to be displayed from the top in the argument
# Importing test data



test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

test.head(6) # Mention no of rows to be displayed from the top in the argument
train.info()
train.describe().transpose()
# In training set



for i in range(train.shape[1]):

    print(train.columns[i],"-",train.iloc[:,i].isnull().sum())
# In test set



for i in range(test.shape[1]):

    print(test.columns[i],"-",test.iloc[:,i].isnull().sum())
fig = plt.figure(figsize=(15,8))

sns.distplot(train["SalePrice"],bins=26,color="brown")

sns.set_style("white")

sns.set_context("poster",font_scale=2)

plt.tight_layout()
#skewness 



print("Skewness: " + str(train['SalePrice'].skew()))
style.use('ggplot')

sns.set_style('whitegrid')

plt.subplots(figsize = (30,20))

 



mask = np.zeros_like(train.corr(), dtype=np.bool)

mask[np.triu_indices_from(mask)] = True





sns.heatmap(train.corr(), 

            cmap=sns.diverging_palette(20, 220, n=200), 

            mask = mask, 

            annot=True, 

            center = 0, 

            cbar="coolwarm",

           );

plt.tight_layout()
train.corr()["SalePrice"].sort_values(ascending=False)[1:]
fig = plt.figure(figsize=(15,8))

sns.scatterplot(x="OverallQual",y="SalePrice",data=train)

sns.set_style("whitegrid")

sns.set_context("poster",font_scale=2)

plt.tight_layout()
fig = plt.figure(figsize=(15,8))

sns.scatterplot(x="GrLivArea",y="SalePrice",data=train)

sns.set_style("whitegrid")

sns.set_context("notebook",font_scale=1.5)

plt.tight_layout()
fig = plt.figure(figsize=(15,8))

sns.scatterplot(x="GarageArea",y="SalePrice",data=train)

sns.set_style("whitegrid")

sns.set_context("notebook",font_scale=2)

plt.tight_layout()
## Deleting those two values with outliers.

train = train[train.GrLivArea < 4500]

train.reset_index(drop = True, inplace = True)



previous_train = train.copy()

print(train.shape)
train["SalePrice"] = np.log1p(train["SalePrice"])

train.drop(columns=['Id'],axis=1, inplace=True)

test.drop(columns=['Id'],axis=1, inplace=True)



## Saving the target values in "y_train". 

y = train['SalePrice'].reset_index(drop=True)







# getting a copy of train

previous_train = train.copy()
## Combining train and test datasets together so that we can do all the work at once. 

all_data = pd.concat((train, test)).reset_index(drop = True)

## Dropping the target variable. 

all_data.drop(['SalePrice'], axis = 1, inplace = True)

# count of missing values in each feature

for i in range(all_data.shape[1]):

    print(all_data.columns[i],"-",all_data.iloc[:,i].isnull().sum())
missing_val_col = ["Alley", 

                   "PoolQC", 

                   "MiscFeature",

                   "Fence",

                   "FireplaceQu",

                   "GarageType",

                   "GarageFinish",

                   "GarageQual",

                   "GarageCond",

                   'BsmtQual',

                   'BsmtCond',

                   'BsmtExposure',

                   'BsmtFinType1',

                   'BsmtFinType2',

                   'MasVnrType']



for i in missing_val_col:

    all_data[i] = all_data[i].fillna('None')
## In the following features the null values are there for a purpose, so we replace them with "0"

missing_val_col2 = ['BsmtFinSF1',

                    'BsmtFinSF2',

                    'BsmtUnfSF',

                    'TotalBsmtSF',

                    'BsmtFullBath', 

                    'BsmtHalfBath', 

                    'GarageYrBlt',

                    'GarageArea',

                    'GarageCars',

                    'MasVnrArea']



for i in missing_val_col2:

    all_data[i] = all_data[i].fillna(0)

    

## Replaced all missing values in LotFrontage by imputing the median value of each neighborhood. 

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))
## Zoning class are given in numerical; therefore converted to categorical variables. 

all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str) 
all_data['Functional'] = all_data['Functional'].fillna('Typ') 

all_data['Utilities'] = all_data['Utilities'].fillna('AllPub') 

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) 

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA") 

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr") 
# count of missing values in each feature



sum = 0

for i in range(all_data.shape[1]):

    sum = sum + all_data.iloc[:,i].isnull().sum()

print(sum)    
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)



skewed_feats
fig = plt.figure(figsize=(15,8))

sns.distplot(train["1stFlrSF"],bins=26,color="brown")

sns.set_style("white")

sns.set_context("poster",font_scale=2)

plt.tight_layout()
## Fixing Skewed features 

def fixing_skewness(df):

    """

    This function takes in a dataframe and return fixed skewed dataframe

    """

    ## Import necessary modules 

    from scipy.stats import skew

    from scipy.special import boxcox1p

    from scipy.stats import boxcox_normmax

    

    ## Getting all the data that are not of "object" type. 

    numeric_feats = df.dtypes[df.dtypes != "object"].index



    # Check the skew of all numerical features

    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

    high_skew = skewed_feats[abs(skewed_feats) > 0.5]

    skewed_features = high_skew.index



    for feat in skewed_features:

        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))



fixing_skewness(all_data)
sns.distplot(all_data['1stFlrSF']);
all_data['TotalSF'] = (all_data['TotalBsmtSF'] 

                       + all_data['1stFlrSF'] 

                       + all_data['2ndFlrSF'])



all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']



all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] 

                                 + all_data['BsmtFinSF2'] 

                                 + all_data['1stFlrSF'] 

                                 + all_data['2ndFlrSF']

                                )

                                 



all_data['Total_Bathrooms'] = (all_data['FullBath'] 

                               + (0.5 * all_data['HalfBath']) 

                               + all_data['BsmtFullBath'] 

                               + (0.5 * all_data['BsmtHalfBath'])

                              )

                               



all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] 

                              + all_data['3SsnPorch'] 

                              + all_data['EnclosedPorch'] 

                              + all_data['ScreenPorch'] 

                              + all_data['WoodDeckSF']

                             )
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

all_data.shape
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
final_features = pd.get_dummies(all_data).reset_index(drop=True)

final_features.shape
X = final_features.iloc[:len(y), :]



X_sub = final_features.iloc[len(y):, :]
outliers = [30, 88, 462, 631, 1322]

X = X.drop(X.index[outliers])

y = y.drop(y.index[outliers])
def overfit_reducer(df):

    """

    This function takes in a dataframe and returns a list of features that are overfitted.

    """

    overfit = []

    for i in df.columns:

        counts = df[i].value_counts()

        zeros = counts.iloc[0]

        if zeros / len(df) * 100 > 99.94:

            overfit.append(i)

    overfit = list(overfit)

    return overfit





overfitted_features = overfit_reducer(X)



X = X.drop(overfitted_features, axis=1)

X_sub = X_sub.drop(overfitted_features, axis=1)
X.shape,y.shape, X_sub.shape
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state = 42)
# Ridge



alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]

temp_rss = {}

temp_mse = {}

for i in alpha_ridge:

    

    ridge = Ridge(alpha= i, normalize=True)

    

    ridge.fit(X_train, y_train)



    y_pred = ridge.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    temp_mse[i] = mse

for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))    
# Lasso



temp_mse = {}

for i in alpha_ridge:

     

    lasso_reg = Lasso(alpha= i, normalize=True)

    

    lasso_reg.fit(X_train, y_train)

    

    y_pred = lasso_reg.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    

    temp_mse[i] = mse

for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
# Elastic Net



from sklearn.linear_model import ElasticNet



temp_mse = {}

for i in alpha_ridge:

 

    lasso_reg = ElasticNet(alpha= i, normalize=True)

    

    lasso_reg.fit(X_train, y_train)

    

    y_pred = lasso_reg.predict(X_test)



    mse = mean_squared_error(y_test, y_pred)

    

    temp_mse[i] = mse

for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):

    print("%s: %s" % (key, value))
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
# Ridge, Lasso and Elastic Net

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 

                                              alphas=alphas2, 

                                              random_state=42, 

                                              cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))  
score = cv_rmse(ridge)

print("Ridge:" , score.mean(), score.std())



score = cv_rmse(lasso)

print("Lasso:" , score.mean(), score.std())



score = cv_rmse(elasticnet)

print("ElasticNet:" , score.mean() , score.std())
# SVR



svr = SVR()

parameters = {'C':[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30],'epsilon':[0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009],'gamma':[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]}

rs= RobustScaler()

X_rs = rs.fit_transform(X)

svr_reg = RandomizedSearchCV(svr,parameters,scoring="neg_mean_squared_error",cv = 5,n_iter = 100,verbose =3,n_jobs =-1)

svr_reg.fit(X_rs,y)
svr_reg.best_params_
# SVR



svr = make_pipeline(RobustScaler(), SVR(C= 28, epsilon= 0.001, gamma=0.0002,))

score = cv_rmse(svr)

print("SVR:" , score.mean() , score.std())
# LGBMRegressor



lgbm = LGBMRegressor(objective='regression',random_state=42)

parameters={'num_leaves':[1,3,4,5,6,8,10],'learning_rate':[0.001,0.01,0.02,0.03,0.04,0.05,0.06],'n_estimators':[500,1000,3000,5000,10000],'max_bin':[100,200,300,400,500],'bagging_fraction':[0.25,0.50,0.75],'bagging_freq':[3,4,5,6,7], 'bagging_seed':[5,6,7,8,9],'feature_fraction':[0.1,0.2,0.3,0.4,0.5,0.6],'feature_fraction_seed':[5,6,7,8,9]}

lgbm_reg = RandomizedSearchCV(lgbm,parameters,scoring="neg_mean_squared_error",cv = 5,n_iter = 100,verbose =3,n_jobs =-1)

lgbm_reg.fit(X,y)
lgbm_reg.best_params_
np.sqrt(-lgbm_reg.best_score_)
# LGBMRegressor



lgbm = LGBMRegressor(objective='regression',random_state=42,num_leaves=10,

  n_estimators=3000,

  max_bin= 300,

  learning_rate= 0.01,

  feature_fraction_seed=5,

  feature_fraction=0.3,

  bagging_seed=8,

  bagging_freq=4,

  bagging_fraction=0.75)
# Xgboost



xgb = XGBRegressor(random_state=42,learning_rate=0.01)

parameters={'n_estimators':[3000,3500,3250,3750,4000,5000],'max_depth':[1,2,3,4,5,6],'min_child_weight':[1,3,5,7],'gamma':[0.0,0.1,0.2,0.3],'colsample_bytree':[0.3,0.6,0.7]}

xgb_reg = RandomizedSearchCV(xgb,parameters,scoring="neg_mean_squared_error",cv = 5,n_iter = 5,verbose =3,n_jobs =-1)

xgb_reg.fit(X,y)
xgb_reg.best_params_
np.sqrt(-xgb_reg.best_score_)
# Xgboost



xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3500,

                                     max_depth=5, min_child_weight=5,

                                     gamma=0.0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27)
# Stacking of regression model

stack_reg = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, xgboost, lgbm ),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
stack_model = stack_reg.fit(np.array(X), np.array(y))

print(1)



elastic_model = elasticnet.fit(X, y)

print(2)



lasso_model = lasso.fit(X, y)

print(3)



ridge_model = ridge.fit(X, y)

print(4)



svr_model = svr.fit(X, y)

print(5)



xgb_model = xgboost.fit(X, y)

print(6)



lgbm_model = lgbm.fit(X, y)

print(7)
def blend_models(X):

    return ((0.1 * elastic_model.predict(X)) + \

            (0.05 * lasso_model.predict(X)) + \

            (0.2 * ridge_model.predict(X)) + \

            (0.1 * svr_model.predict(X)) + \

            (0.15 * xgb_model.predict(X)) + \

            (0.1 * lgbm_model.predict(X)) + \

            (0.3 * stack_model.predict(np.array(X))))
print(rmsle(y, blend_models(X)))
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = np.floor(np.expm1(blend_models(X_sub)))
sub_1 = pd.read_csv('../input/top-submission1/submission 1.csv')

sub_2 = pd.read_csv('../input/top-submission2/submission 2.csv')

sub_3 = pd.read_csv('../input/top-submission3/submission 3.csv')

submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models(X_sub)))) + 

                                (0.25 * sub_1.iloc[:,1]) + 

                                (0.25 * sub_2.iloc[:,1]) + 

                                (0.25 * sub_3.iloc[:,1]))
q1 = submission['SalePrice'].quantile(0.005)

q2 = submission['SalePrice'].quantile(0.995)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission_n.csv", index=False)