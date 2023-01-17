# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.gaussian_process.kernels import ExpSineSquared

from sklearn.preprocessing import LabelEncoder, RobustScaler

from scipy.stats import skew

from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV,learning_curve

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

%matplotlib inline
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train.shape
test.shape
train.info()
train.describe()
train['SalePrice'].dropna().mean()
sns.distplot(train['SalePrice'])
sns.distplot(train['YearBuilt'])
sns.lmplot(x='GrLivArea', y='SalePrice', data=train)
train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)]
train.drop([523,1298], inplace=True)
sns.lmplot(x='GrLivArea', y='SalePrice', data=train)
train_length = len(train)

combined = pd.concat([train, test])
sns.heatmap(combined.isnull())
combined.isnull().sum().sort_values(ascending=False)[:40]
combined['PoolQC'].fillna('No Pool', inplace=True)

combined['MiscFeature'].fillna('None', inplace=True)

combined['Alley'].fillna('No alley access', inplace=True)

combined['Fence'].fillna('No Fence', inplace=True)

combined['FireplaceQu'].fillna('No Fireplace', inplace=True)
combined["LotFrontage"] = combined.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.dropna().median()))
combined[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = combined[['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']].fillna('No Garage')
combined[['GarageYrBlt', 'GarageArea', 'GarageCars']] = combined[['GarageYrBlt', 'GarageArea', 'GarageCars']].fillna(0)
combined[['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']] = combined[['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtCond', 'BsmtQual']].fillna('No Basement')
combined['MasVnrArea'] = combined['MasVnrArea'].fillna(0)

combined['MasVnrType'] = combined['MasVnrType'].fillna('None')
combined['Electrical'] = combined['Electrical'].fillna(combined['Electrical'].mode()[0])

combined['MSZoning'] = combined['MSZoning'].fillna(combined['MSZoning'].mode()[0])

combined['Functional'] = combined['Functional'].fillna(combined['Functional'].mode()[0])

combined['Exterior1st'] = combined['Exterior1st'].fillna(combined['Exterior1st'].mode()[0])

combined['Exterior2nd'] = combined['Exterior2nd'].fillna(combined['Exterior2nd'].mode()[0])

combined['KitchenQual'] = combined['KitchenQual'].fillna(combined['KitchenQual'].mode()[0])

combined['SaleType'] = combined['SaleType'].fillna(combined['SaleType'].mode()[0])
combined[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = combined[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']].fillna(0) 
sns.countplot(x='Utilities', data=combined)
combined.drop('Utilities', axis=1, inplace=True)
combined['TotalBsmtSF'].fillna(0, inplace=True)
combined.isnull().sum().sort_values(ascending=False)[:20]
train["SalePrice"] = train["SalePrice"].map(lambda i: np.log1p(i))
sns.distplot(train['SalePrice'])
numeric_columns = []

categorical_columns = []

for column in combined.columns:

    if(combined[column].dtype == np.object):

        categorical_columns.append(column)

    else :

        numeric_columns.append(column)
len(categorical_columns)
for column in categorical_columns:

    combined[column] = LabelEncoder().fit_transform(combined[column])
len(numeric_columns)
skewed_columns = combined[numeric_columns].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_columns = skewed_columns.apply(abs)
for column in skewed_columns[skewed_columns > 0.75].index:

    combined[column] = combined[column].apply(lambda x : np.log1p(x))
train = combined[:train_length]

test = combined[train_length:].drop('SalePrice', axis=1)
train.shape
test.shape
X = train.drop('SalePrice', axis=1)

y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
KRR = KernelRidge()

GBR = GradientBoostingRegressor()

XGB = XGBRegressor()

LGBM = LGBMRegressor()

ENET =  ElasticNet()

LASS =  Lasso()
models = [KRR, GBR, XGB, LGBM, ENET, LASS]
k_folds = KFold(5, shuffle=True, random_state=42)



def cross_val_rmse(model):

    return np.sqrt(-1*cross_val_score(model, X, y,scoring="neg_mean_squared_error",cv=k_folds))
corss_val_score = []

for model in models:

    model_name = model.__class__.__name__

    corss_val_score.append((model_name,cross_val_rmse(model).mean()))
sorted(corss_val_score, key=lambda x : x[1], reverse=True)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
for model in models:

    plot_learning_curve(model,model.__class__.__name__ + " mearning curves",X,y,cv=5)
# lass_param_grid = {

#     'alpha' : [0.001, 0.0005],

#     'random_state':[1,3,42]

# }



# enet_param_grid = {

#     'alpha' : [0.001, 0.0005],

#     'random_state':[1,3,42],

#     'l1_ratio' : [.1,.9] 

# }



# gboost_param_grid ={

#     'n_estimators':[100,3000],

#     'learning_rate': [0.1, 0.05],

#     'max_depth':[4,6],

#     'max_features':['sqrt'],

#     'min_samples_leaf' :[3,9,15],

#     'min_samples_split':[3,10],

#     'loss':['huber'],

#     'random_state':[5,42]

# }



# xgb_param_grid = {

#     'colsample_bytree':[0.1,0.5],

#     'gamma' :[0.01,0.04],

#     'reg_alpha':[0.1,0.5],

#     'reg_lambda':[0.1,0.9],

#     'subsample':[0.1,0.5],

#     'silent':[1],

#     'random_state':[1,7],

#     'nthread':[-1],

#     'learning_rate': [0.1, 0.05],

#     'max_depth': [3,6],

#     'min_child_weight':[1.5,1.4,1.8],

#     'n_estimators': [100,2000]}









# krl_param_grid = {"alpha": [0.1, 0.6],"degree": [2,4], "kernel":['polynomial'], "coef0":[0.5,2.5]}







# lgbm_param_grid = {

#     'n_estimators':[100],

#     'learning_rate': [0.1, 0.05, 0.01],

#     'max_depth':[4,6],

#     'max_leaves':[3,9,17],

# }



# models = [

#     (KernelRidge,krl_param_grid),

#     (XGBRegressor,xgb_param_grid),

#     (GradientBoostingRegressor,gboost_param_grid),

#     (Lasso,lass_param_grid),

#     (ElasticNet,enet_param_grid)

# ]
# best_models = []

# for model, param in models:

#     print("Fitting ", model.__class__.__name__)

#     grid_search = GridSearchCV(model(),

#                                scoring='neg_mean_squared_error',

#                                param_grid=param,

#                                cv=5,

#                                verbose=2,

#                                n_jobs=-1)

#     grid_search.fit(X, y)

#     print(grid_search.best_params_)
GBR =  GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=15, min_samples_split=10, 

                                   loss='huber', random_state =5)







XGB = XGBRegressor(gamma=0.04, learning_rate=0.05, colsample_bytree=0.5,  

                              max_depth=3, 

                             min_child_weight=1.8, n_estimators=2000,

                             reg_alpha=0.5, reg_lambda=0.9,

                             subsample=0.5, silent=1,

                             random_state =7, nthread = -1)

KRR = KernelRidge(kernel='polynomial', alpha=0.6, coef0=2.5, degree=2)



LASS = Lasso(alpha =0.0005, random_state=1)

ENET = ElasticNet( l1_ratio=.9,alpha=0.0005, random_state=3)

best_models = [GBR,XGB,KRR,LASS,ENET]
corss_val_score = []

for model in best_models:

    model_name = model.__class__.__name__

    print("Fitting ",model_name)

    corss_val_score.append((model_name,cross_val_rmse(model).mean()))
sorted(corss_val_score, key=lambda x : x[1], reverse=True)
corss_val_score
total_rmse = sum([x[1] for x in corss_val_score])
weighted_val_score = {}

for k,v in corss_val_score:

    weighted_val_score[k] = round((v/total_rmse)*100)
weighted_val_score
submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
for model in best_models:

    model.fit(X,y)

    y_pred = model.predict(test)

    submission[model.__class__.__name__] = model.predict(test)

    submission[model.__class__.__name__] = submission[model.__class__.__name__].apply(lambda x: np.expm1(x))
submission
submission['AVG2'] = submission[['GradientBoostingRegressor','XGBRegressor', 'Lasso', 'ElasticNet']].mean(axis=1)

weighted_average = submission[['Id', 'AVG2']]

weighted_average.rename(columns={'AVG2':'SalePrice'}, inplace=True)

weighted_average.to_csv("AVG2.csv", index=False)
submission['AVG3'] = submission[['GradientBoostingRegressor','XGBRegressor', 'ElasticNet']].mean(axis=1)

weighted_average = submission[['Id', 'AVG3']]

weighted_average.rename(columns={'AVG3':'SalePrice'}, inplace=True)

weighted_average.to_csv("AVG3.csv", index=False)
submission['AVG4'] = submission[['GradientBoostingRegressor', 'ElasticNet']].mean(axis=1)

weighted_average = submission[['Id', 'AVG4']]

weighted_average.rename(columns={'AVG4':'SalePrice'}, inplace=True)

weighted_average.to_csv("AVG4.csv", index=False)
submission['AVG'] = submission[['GradientBoostingRegressor','XGBRegressor', 'KernelRidge', 'Lasso', 'ElasticNet']].mean(axis=1)

weighted_average = submission[['Id', 'AVG']]

weighted_average.rename(columns={'AVG':'SalePrice'}, inplace=True)

weighted_average.to_csv("AVG.csv", index=False)
submission['weighted_average'] = (submission['GradientBoostingRegressor']*(0.18))+(submission['XGBRegressor']*(0.18))+(submission['KernelRidge']*(0.28))+(submission['Lasso']*(0.18))+(submission['ElasticNet']*(0.18)) 
submission
weighted_average = submission[['Id', 'weighted_average']]

weighted_average.rename(columns={'weighted_average':'SalePrice'}, inplace=True)

weighted_average.to_csv("weighted_average.csv", index=False)
GRB = submission[['Id', 'GradientBoostingRegressor']]

GRB.rename(columns={'GradientBoostingRegressor':'SalePrice'}, inplace=True)

GRB.to_csv("GRB.csv", index=False)
XGBR = submission[['Id', 'XGBRegressor']]

XGBR.rename(columns={'XGBRegressor':'SalePrice'}, inplace=True)

XGBR.to_csv("XGBR.csv", index=False)
KRR = submission[['Id', 'KernelRidge']]

KRR.rename(columns={'KernelRidge':'SalePrice'}, inplace=True)

KRR.to_csv("KRR.csv", index=False)
LASS = submission[['Id', 'Lasso']]

LASS.rename(columns={'Lasso':'SalePrice'}, inplace=True)

LASS.to_csv("LASS.csv", index=False)
ENET = submission[['Id', 'ElasticNet']]

ENET.rename(columns={'ElasticNet':'SalePrice'}, inplace=True)

ENET.to_csv("ENET.csv", index=False)