# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.model_selection import KFold,train_test_split,cross_val_predict,GridSearchCV,cross_val_score

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from sklearn.metrics import mean_squared_error

from sklearn import preprocessing

from scipy.stats import skew  # for some statistics

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax
Train_DF = pd.read_csv("../input/train.csv")

Test_DF = pd.read_csv("../input/test.csv")
Train_DF.drop(['Id'],axis=1,inplace=True)

Test_DF.drop(['Id'],axis=1,inplace=True)

Train_DF.describe()
_Analysis_trainDF = Train_DF

_Analysis_trainDF.head()
#Funtion to identify columns which contains NaN values & those which are having more than 80% NaN values

_EmptyColList = list()

_EmptyColwithNAN = list()

_Maxcount = len(_Analysis_trainDF.iloc[:,0])

for column in _Analysis_trainDF.columns:

    if (_Analysis_trainDF[column].isna().any()):

        _EmptyColList.append(column)

        _TempVar = len(_Analysis_trainDF[_Analysis_trainDF[column].isna()])

        count = (_TempVar/_Maxcount)*100

        if count > 80:

            _EmptyColwithNAN.append(column)
_NullColums = _Analysis_trainDF[_EmptyColList].isna().sum()

_NullColums.sort_values().plot.bar()
Train_DF.drop(_EmptyColwithNAN,axis=1,inplace=True)

Train_DF.reset_index(drop=True, inplace=True)

Train_DF["SalePrice"] = np.log1p(Train_DF["SalePrice"])
_Analysis_trainDF.plot.scatter(x='OverallQual',y='SalePrice'),_Analysis_trainDF.plot.scatter(x='GrLivArea',y='SalePrice')

_Analysis_trainDF.plot.scatter(x='GarageCars',y='SalePrice'),_Analysis_trainDF.plot.scatter(x='GarageArea',y='SalePrice')

_Analysis_trainDF.plot.scatter(x='TotalBsmtSF',y='SalePrice'),_Analysis_trainDF.plot.scatter(x='1stFlrSF',y='SalePrice')

_Analysis_trainDF.plot.scatter(x='FullBath',y='SalePrice'),_Analysis_trainDF.plot.scatter(x='TotRmsAbvGrd',y='SalePrice')

_Analysis_trainDF.plot.scatter(x='YearBuilt',y='SalePrice'),_Analysis_trainDF.plot.scatter(x='GarageYrBlt',y='SalePrice'),
# Removing outliers

Train_DF = Train_DF[(Train_DF['GrLivArea'] < 4500) & (Train_DF['LotArea'] < 50000) & (Train_DF['GarageArea'] < 1220) & (Train_DF['TotalBsmtSF'] < 3000) & (Train_DF['1stFlrSF'] < 4000) & (Train_DF['TotRmsAbvGrd'] < 13) & (Train_DF['YearBuilt'] > 1895)]  

Train_DF.reset_index(drop=True, inplace=True)

Train_DF.shape
y = Train_DF['SalePrice'].reset_index(drop=True)

y.shape
Train_Features = Train_DF.drop(['SalePrice'],axis = 1)

Test_Features = Test_DF

Combined_FeaturesDF = pd.concat([Train_Features, Test_Features]).reset_index(drop=True)

Train_Features.shape, Test_Features.shape, Combined_FeaturesDF.shape
# 2nd floor surface area should be 0 incase of 1 story building

Combined_FeaturesDF[(Combined_FeaturesDF['HouseStyle'] == '1Story') & (Combined_FeaturesDF['2ndFlrSF'] > 0)]['2ndFlrSF'] = 0



# filling below features with most frequent Attribure/values

Combined_FeaturesDF['Electrical'].fillna(Combined_FeaturesDF['Electrical'].mode()[0],inplace=True)

Combined_FeaturesDF['Exterior1st'].fillna(Combined_FeaturesDF['Exterior1st'].mode()[0],inplace=True)

Combined_FeaturesDF['Exterior2nd'].fillna(Combined_FeaturesDF['Exterior2nd'].mode()[0],inplace=True)

Combined_FeaturesDF['Functional'].fillna(Combined_FeaturesDF['Functional'].mode()[0],inplace=True)

Combined_FeaturesDF['KitchenQual'].fillna(Combined_FeaturesDF['KitchenQual'].mode()[0],inplace=True)

Combined_FeaturesDF['MasVnrType'].fillna(Combined_FeaturesDF['MasVnrType'].mode()[0],inplace=True)

Combined_FeaturesDF['SaleType'].fillna(Combined_FeaturesDF['SaleType'].mode()[0],inplace=True)

Combined_FeaturesDF['Utilities'].fillna(Combined_FeaturesDF['Utilities'].mode()[0],inplace=True)

Combined_FeaturesDF['BsmtExposure'].fillna(Combined_FeaturesDF['BsmtExposure'].mode()[0],inplace=True)

Combined_FeaturesDF['FireplaceQu'].fillna(0,inplace=True)



#Idea is that similar MSSubClasses will have similar MSZoning

Combined_FeaturesDF['MSZoning'] = Combined_FeaturesDF.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



# For those which are having no Bsmt area we can fill it with None.

for col in ('BsmtCond','BsmtFinType1','BsmtFinType2','BsmtQual'):

    Combined_FeaturesDF[col] = Combined_FeaturesDF[col].fillna('None')

# For those which are having no garage area we can fill it with None then we will encode it with some value

for col in ('GarageCond','GarageFinish','GarageQual','GarageType'):

    Combined_FeaturesDF[col] = Combined_FeaturesDF[col].fillna('None')
Combined_FeaturesDF['LotFrontage'] = Combined_FeaturesDF.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))



#below we are setting numeric features having Nan to 0

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics = []

for i in Combined_FeaturesDF.columns:

    if Combined_FeaturesDF[i].dtype in numeric_dtypes:

        numerics.append(i)

Combined_FeaturesDF.update(Combined_FeaturesDF[numerics].fillna(0))

numerics[1:100]
# Adding new features.

Combined_FeaturesDF['YrBltAndRemod']=Combined_FeaturesDF['YearBuilt']+Combined_FeaturesDF['YearRemodAdd']

Combined_FeaturesDF['TotalSF']=Combined_FeaturesDF['TotalBsmtSF'] + Combined_FeaturesDF['1stFlrSF'] + Combined_FeaturesDF['2ndFlrSF']



Combined_FeaturesDF['Total_sqr_footage'] = (Combined_FeaturesDF['BsmtFinSF1'] + Combined_FeaturesDF['BsmtFinSF2'] +

                                 Combined_FeaturesDF['1stFlrSF'] + Combined_FeaturesDF['2ndFlrSF'])



Combined_FeaturesDF['Total_Bathrooms'] = (Combined_FeaturesDF['FullBath'] + (0.5 * Combined_FeaturesDF['HalfBath']) +

                               Combined_FeaturesDF['BsmtFullBath'] + (0.5 * Combined_FeaturesDF['BsmtHalfBath']))



Combined_FeaturesDF['Total_porch_sf'] = (Combined_FeaturesDF['OpenPorchSF'] + Combined_FeaturesDF['3SsnPorch'] +

                              Combined_FeaturesDF['EnclosedPorch'] + Combined_FeaturesDF['ScreenPorch'] +

                              Combined_FeaturesDF['WoodDeckSF'])
#Adding few more features.

Combined_FeaturesDF['haspool'] = Combined_FeaturesDF['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

Combined_FeaturesDF['has2ndfloor'] = Combined_FeaturesDF['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)

Combined_FeaturesDF['hasgarage'] = Combined_FeaturesDF['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

Combined_FeaturesDF['hasbsmt'] = Combined_FeaturesDF['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

Combined_FeaturesDF['hasfireplace'] = Combined_FeaturesDF['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
 ### Fill the remaining columns as **None**

objects = []

for i in Combined_FeaturesDF.columns:

    if Combined_FeaturesDF[i].dtype == object:

        objects.append(i)

Combined_FeaturesDF.update(Combined_FeaturesDF[objects].fillna('None'))

print(objects)
#Removing Skewness of numeric columns in dataset

numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

numerics2 = []

for i in Combined_FeaturesDF.columns:

    if Combined_FeaturesDF[i].dtype in numeric_dtypes:

        numerics2.append(i)

skew_features = Combined_FeaturesDF[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)



high_skew = skew_features[skew_features > 0.5]

skew_index = high_skew.index



for i in skew_index:

    Combined_FeaturesDF[i] = boxcox1p(Combined_FeaturesDF[i], boxcox_normmax(Combined_FeaturesDF[i] + 1))
# We are putting categorical features in list.



cols = Combined_FeaturesDF.columns

num_cols = Combined_FeaturesDF._get_numeric_data().columns

num_cols



_ListOf_Cat_column = list(set(cols) - set(num_cols))

_ListOf_Cat_column

#Comment out for now , let see what is the when we are trying using get_dummies

'''le = preprocessing.LabelEncoder()

# apply le on categorical feature columns

Combined_FeaturesDF[_ListOf_Cat_column] = Combined_FeaturesDF[_ListOf_Cat_column].apply(lambda col: le.fit_transform(col))

Combined_FeaturesDF.head(10)'''



# One Hot Encoding for categorical features

final_features = pd.get_dummies(Combined_FeaturesDF).reset_index(drop=True)

final_features.shape
final_features.head()
C_mat = _Analysis_trainDF.corr()

fig = plt.figure(figsize = (15,15))



sns.heatmap(C_mat, vmax = .8, square = True, annot=True)

plt.show()

C_mat.OverallQual.sort_values(ascending=False)
C_mat.SalePrice.sort_values(ascending=False),C_mat.SalePrice.sort_values(ascending=False).plot.hist()
# creating data for fitting model

X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]

X.shape, y.shape, X_sub.shape
import xgboost as xgb

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

import matplotlib.pyplot as pyplot

from mlxtend.regressor import StackingCVRegressor

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.svm import SVR
# defining error functions for handy use. 

kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, T=X):

    rmse = np.sqrt(-cross_val_score(model, T, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, alphas=alphas2, random_state=42, cv=kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4,

                                       learning_rate=0.01, 

                                       n_estimators=5000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2,

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3000,

                                     max_depth=3, min_child_weight=0,

                                     gamma=0, subsample=0.7,

                                     colsample_bytree=0.7,

                                     objective='reg:linear', nthread=-1,

                                     scale_pos_weight=1, seed=27,

                                     reg_alpha=0.00006)
score = cv_rmse(ridge)

print(score)

score = cv_rmse(lasso)

print(score)

score = cv_rmse(elasticnet)

print(score)

score = cv_rmse(gbr)

print(score)

score = cv_rmse(lightgbm)

print(score)

score = cv_rmse(xgboost)

print(score)
#In order to avoid overfitting. we are using StackingCVRegressor

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgboost, lightgbm),

                                meta_regressor=xgboost,

                                use_features_in_secondary=True)
#Start Fitting

print('Start fitting models')

#stack_gen.fit(X_train,y_train)

print('elasticnet')

elastic_model_full_data = elasticnet.fit(X, y)



print('Lasso')

lasso_model_full_data = lasso.fit(X, y)



print('Ridge')

ridge_model_full_data = ridge.fit(X, y)



stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print('stack_gen_model',stack_gen_model)



GradiantBoosting_model = gbr.fit(X, y)

print('GradiantBoosting_model',GradiantBoosting_model)



XGBoosting_model = xgboost.fit(X, y)

print('XGBoosting_model',XGBoosting_model)



LightGBM_model = lightgbm.fit(X, y)

print('LightGBM_model',LightGBM_model)
# Assigning different weight of models in order to create final one depending on scores of each model

def combine_models_predict(X):

    return ((0.15 * GradiantBoosting_model.predict(X)) + \

            (0.2 * XGBoosting_model.predict(X)) + \

            (0.1 * ridge.predict(X)) + \

            (0.05 * lasso.predict(X)) + \

            (0.1 * elasticnet.predict(X)) + \

            (0.1 * LightGBM_model.predict(X)) + \

            (0.3 * stack_gen_model.predict(np.array(X))))
print('Predict submission')

submission = pd.read_csv("../input/sample_submission.csv")

submission['SalePrice'] = (np.expm1(combine_models_predict(X_sub)))

submission.to_csv("sample_submission.csv",index=False)