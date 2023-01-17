# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from scipy.stats import norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax

from sklearn.preprocessing import StandardScaler

from scipy import stats



import warnings

warnings.filterwarnings('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Read files

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#print(train)
################

# prepare data #

################



#Save ID

train_id = train["Id"]

test_id = test["Id"]



#Drop ID columns (no need for prediction)

train.drop("Id", axis=1, inplace= True)

test.drop("Id", axis=1, inplace= True)

#Describe Data columns

#print (train.columns)

#print(test.columns)

#print(train.shape,test.shape)

#for i in train.columns:

#    if i not in test.columns:

#        print("Missing colums : |%s|" % (i,))

#train.describe()

train['SalePrice'].describe()

sns.distplot(train['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())



sns.distplot(np.log1p(train.SalePrice ));
# Apply log transformation on sale price to 

# get Normal distribution

train_salePrice = train.SalePrice 

train.SalePrice = np.log1p(train.SalePrice )

# New prediction

y_train = train.SalePrice.values

y_train_orig = train.SalePrice
data_features = pd.concat((train, test), sort=True).reset_index(drop=True)

print(data_features.shape)



# Missing data in train

data_features_na = data_features.isnull().sum()

data_features_na = data_features_na[data_features_na>0]

data_features_na.sort_values(ascending=False)
str_vars = ['MSSubClass','YrSold','MoSold']

for var in str_vars:

    data_features[var] = data_features[var].apply(str)

    

#__________________________________________________________________________________    

#Version 1 (Most popular value!)

#common_vars = ['Exterior1st','Exterior2nd','SaleType','Electrical','KitchenQual']

#for var in common_vars:

#    data_features[var] = data_features[var].fillna(data_features[var].mode()[0])   

#__________________________________________________________________________________

#Version 2 (empty data ~> worst case)

common_vars = ['Exterior1st','Exterior2nd']

for var in common_vars:

    data_features[var] = data_features[var].fillna('Other')   



common_vars = ['SaleType']

for var in common_vars:

    data_features[var] = data_features[var].fillna('Oth')   



common_vars = ['Electrical']

for var in common_vars:

    data_features[var] = data_features[var].fillna('FuseP')   



common_vars = ['KitchenQual']

for var in common_vars:

    data_features[var] = data_features[var].fillna('Po')   

#__________________________________________________________________________________



#Version 0

#data_features['MSZoning'] = data_features.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))    

#Version 1

#data_features['MSZoning'] = data_features['MSZoning'].fillna(data_features['MSZoning'].mode()[0])   

#Version 2

data_features['MSZoning'] = data_features['MSZoning'].fillna('I')   



# Replacing missing data with None

for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond','BsmtQual',

            'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',"PoolQC"

           ,'Alley','Fence','MiscFeature','FireplaceQu','MasVnrType','Utilities']:

    data_features[col] = data_features[col].fillna('None')

    

# Replacing missing data with 0 (Since No garage = no cars in such garage.)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','MasVnrArea','BsmtFinSF1','BsmtFinSF2'

           ,'BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BsmtUnfSF','TotalBsmtSF'):

    data_features[col] = data_features[col].fillna(0)



# group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

#Version 1

#data_features['LotFrontage'] = data_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

#Version 2

data_features['LotFrontage'] = data_features.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.min()))



print('Features size:', data_features.shape)



# data description says NA means typical

data_features['Functional'] = data_features['Functional'].fillna('Typ')
#data_features[var]



#missing data check

#total = data_features.isnull().sum().sort_values(ascending=False)

#percent = (data_features.isnull().sum()/data_features.isnull().count()).sort_values(ascending=False)

#missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#missing_data.head(10)
# Differentiate numerical features (minus the target) and categorical features

categorical_features = data_features.select_dtypes(include=['object']).columns

print(categorical_features)

numerical_features = data_features.select_dtypes(exclude = ["object"]).columns

print(numerical_features)



print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))

feat_num = data_features[numerical_features]

feat_cat = data_features[categorical_features]
# Plot skew value for each numerical value

from scipy.stats import skew 

skewness = feat_num.apply(lambda x: skew(x))

skewness.sort_values(ascending=False)
skewness = skewness[abs(skewness) > 0.5]

print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

print("Mean skewnees: {}".format(np.mean(skewness)))



from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feat in skewed_features:

    feat_num[feat] = boxcox1p(feat_num[feat], boxcox_normmax(feat_num[feat] + 1))

    data_features[feat] = boxcox1p(data_features[feat], boxcox_normmax(data_features[feat] + 1))

    

    

from scipy.stats import skew 

skewness.sort_values(ascending=False)
skewness = feat_num.apply(lambda x: skew(x))

skewness = skewness[abs(skewness) > 0.5]



print("There are {} skewed numerical features after Box Cox transform".format(skewness.shape[0]))

print("Mean skewnees: {}".format(np.mean(skewness)))

skewness.sort_values(ascending=False)
# Calculating totals before droping less significant columns



#  Adding total sqfootage feature 

data_features['TotalSF']=data_features['TotalBsmtSF'] + data_features['1stFlrSF'] + data_features['2ndFlrSF']

#  Adding total bathrooms feature

data_features['Total_Bathrooms'] = (data_features['FullBath'] + (0.5 * data_features['HalfBath']) +

                               data_features['BsmtFullBath'] + (0.5 * data_features['BsmtHalfBath']))

#  Adding total porch sqfootage feature

data_features['Total_porch_sf'] = (data_features['OpenPorchSF'] + data_features['3SsnPorch'] +

                              data_features['EnclosedPorch'] + data_features['ScreenPorch'] +

                              data_features['WoodDeckSF'])





data_features['haspool'] = data_features['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

data_features['hasgarage'] = data_features['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

data_features['hasbsmt'] = data_features['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data_features['hasfireplace'] = data_features['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

# data_features['Super_quality'] = OverallQual * 

# vars = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'FullBath']
# Not normaly distributed can not be normalised and has no central tendecy

data_features = data_features.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF'], axis=1)

# data_features = data_features.drop(['MasVnrArea', 'OpenPorchSF', 'WoodDeckSF', 'BsmtFinSF1','2ndFlrSF',

#                          'PoolArea','3SsnPorch','LowQualFinSF','MiscVal','BsmtHalfBath','ScreenPorch',

#                          'ScreenPorch','KitchenAbvGr','BsmtFinSF2','EnclosedPorch','LotFrontage'

#                          ,'BsmtUnfSF','GarageYrBlt'], axis=1)



print('data_features size:', data_features.shape)

train = data_features.iloc[:len(y_train), :]

test = data_features.iloc[len(y_train):, :]

print(['Train data shpe: ',train.shape,'Prediction on (Sales price) shape: ', y_train.shape,'Test shape: ', test.shape])
vars = data_features.columns

# vars = numerical_features

figures_per_time = 4

count = 0 

y = y_train

for var in vars:

    x = train[var]

#     print(y.shape,x.shape)

    plt.figure(count//figures_per_time,figsize=(25,5))

    plt.subplot(1,figures_per_time,np.mod(count,4)+1)

    plt.scatter(x, y);

    plt.title('f model: T= {}'.format(var))

    count+=1
# Removes outliers 

# outliers = [30, 88, 462, 631, 1322]

# train = train.drop(train.index[outliers])

y_train = train['SalePrice']
# vars_box = ['OverallQual','YearBuilt','BedroomAbvGr']

vars_box = feat_cat

for var in vars_box:

    data = pd.concat([train['SalePrice'], train[var]], axis=1)

    f, ax = plt.subplots(figsize=(8, 6))

    fig = sns.boxplot(x=var, y="SalePrice", data=data)
# Complete numerical correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=1, square=True);
# saleprice correlation matrix

corr_num = 15 #number of variables for heatmap

cols_corr = corrmat.nlargest(corr_num, 'SalePrice')['SalePrice'].index

corr_mat_sales = np.corrcoef(train[cols_corr].values.T)

sns.set(font_scale=1.25)

f, ax = plt.subplots(figsize=(12, 9))

hm = sns.heatmap(corr_mat_sales, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 7}, yticklabels=cols_corr.values, xticklabels=cols_corr.values)

plt.show()
# pair plots for variables with largest correlation

var_num = 8

vars = cols_corr[0:var_num]



sns.set()

sns.pairplot(train[vars], size = 2.5)

plt.show();
data_features = data_features.drop("SalePrice", axis = 1)

final_features = pd.get_dummies(data_features)



print(final_features.shape)

X = final_features.iloc[:len(y), :]

X_test = final_features.iloc[len(y):, :]

X.shape, y_train.shape, X_test.shape





print(X.shape,y_train.shape,X_test.shape)
# Removes colums where the threshold of zero's is (> 99.95), means has only zero values 

overfit = []

for i in X.columns:

    counts = X[i].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(X) * 100 > 99.95:

        overfit.append(i)



overfit = list(overfit)

overfit.append('MSZoning_C (all)')



X = X.drop(overfit, axis=1).copy()

X_test = X_test.drop(overfit, axis=1).copy()



print(X.shape,y_train.shape,X_test.shape)
from datetime import datetime

from sklearn.preprocessing import RobustScaler

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error , make_scorer

from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LinearRegression



from sklearn.ensemble import GradientBoostingRegressor

# from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

from sklearn.linear_model import LinearRegression



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)



# model scoring and validation function

def cv_rmse(model, X=X):

    rmse = np.sqrt(-cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=kfolds))

    return (rmse)



# rmsle scoring function

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
lightgbm = LGBMRegressor(objective='regression', 

                                       num_leaves=4, #was 3

                                       learning_rate=0.01, 

                                       n_estimators=8000,

                                       max_bin=200, 

                                       bagging_fraction=0.75,

                                       bagging_freq=5, 

                                       bagging_seed=7,

                                       feature_fraction=0.2, # 'was 0.2'

                                       feature_fraction_seed=7,

                                       verbose=-1,

                                       )



# xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,

#                                      max_depth=3, min_child_weight=0,

#                                      gamma=0, subsample=0.7,

#                                      colsample_bytree=0.7,

#                                      objective='reg:linear', nthread=-1,

#                                      scale_pos_weight=1, seed=27,

#                                      reg_alpha=0.00006)







# setup models hyperparameters using a pipline

# The purpose of the pipeline is to assemble several steps that can be cross-validated together, while setting different parameters.

# This is a range of values that the model considers each time in runs a CV

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]

alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]









# Kernel Ridge Regression : made robust to outliers

ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))



# LASSO Regression : made robust to outliers

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 

                    alphas=alphas2,random_state=42, cv=kfolds))



# Elastic Net Regression : made robust to outliers

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 

                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))





stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, lightgbm),

                                meta_regressor=elasticnet,

                                use_features_in_secondary=True)



# store models, scores and prediction values 

models = {'Ridge': ridge,

          'Lasso': lasso, 

          'ElasticNet': elasticnet,

          'lightgbm': lightgbm}

#           'xgboost': xgboost}

predictions = {}

scores = {}
for name, model in models.items():

    

    model.fit(X, y)

    predictions[name] = np.expm1(model.predict(X))

    

    score = cv_rmse(model, X=X)

    scores[name] = (score.mean(), score.std())
# get the performance of each model on training data(validation set)

scoreList = []

print('---- Score with CV_RMSE-----')

score = cv_rmse(ridge)

print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

scoreList.append(1 - score.mean())



score = cv_rmse(lasso)

print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

scoreList.append(1 - score.mean())



score = cv_rmse(elasticnet)

print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

scoreList.append(1 - score.mean())



score = cv_rmse(lightgbm)

print("lightgbm score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

scoreList.append(1 - score.mean())



score = cv_rmse(stack_gen)

print("stack_gen score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

scoreList.append(1 - score.mean())



# score = cv_rmse(xgboost)

# print("xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))





#Fit the training data X, y

print('----START Fit----',datetime.now())

print('Elasticnet')

elastic_model = elasticnet.fit(X, y)

print('Lasso')

lasso_model = lasso.fit(X, y)

print('Ridge')

ridge_model = ridge.fit(X, y)

print('lightgbm')

lgb_model_full_data = lightgbm.fit(X, y)

print('stack_gen')

stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

# print('xgboost')

# xgb_model_full_data = xgboost.fit(X, y)
def blend_models_predict(X):

    return ((0.25  * elastic_model.predict(X)) + \

            (0.25 * lasso_model.predict(X)) + \

            (0.2 * ridge_model.predict(X)) + \

            (0.10 * lgb_model_full_data.predict(X)) + \

#             (0.1 * xgb_model_full_data.predict(X)) + \

            (0.2 * stack_gen_model.predict(np.array(X))))



def blend_models_predict_ABL(X, _scoreList):

    

    print("scoreList~>|%s|" % (_scoreList,))

    #ScoreSum = 0

    #for i in range(len(_scoreList)):

    #    ScoreSum += _scoreList[i]

    ScoreSum = sum(_scoreList)

    print("ScoreSum~>|%s|" % (ScoreSum,))    

    return ((scoreList[2]/ScoreSum  * elastic_model.predict(X)) + \

            (scoreList[1]/ScoreSum * lasso_model.predict(X)) + \

            (scoreList[0]/ScoreSum * ridge_model.predict(X)) + \

            (scoreList[3]/ScoreSum * lgb_model_full_data.predict(X)) + \

#             (0.1 * xgb_model_full_data.predict(X)) + \

            (scoreList[4]/ScoreSum* stack_gen_model.predict(np.array(X))))
print('RMSLE score on train data:')

print(rmsle(y, blend_models_predict(X)))



print('RMSLE score on train data: (blend_models_predict2)')

print(rmsle(y, blend_models_predict_ABL(X, scoreList)))
print('Predict submission')

submission = pd.read_csv("../input/sample_submission.csv")

submission.iloc[:,1] = (np.expm1(blend_models_predict_ABL(X_test, scoreList)))
#### q1 = submission['SalePrice'].quantile(0.0042)

# q2 = submission['SalePrice'].quantile(0.99)

# # Quantiles helping us get some extreme values for extremely low or high values 

# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)

# submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)

submission.to_csv("submission.csv", index=False)