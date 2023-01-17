import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns    

import math



from scipy import stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from sklearn.preprocessing import  LabelEncoder

from sklearn.linear_model import ElasticNetCV, LassoCV, BayesianRidge, RidgeCV

from sklearn.svm import SVR

from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from mlxtend.regressor import StackingCVRegressor



import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")



train.columns
train['SalePrice'].describe()
corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index, inplace = True)
fig, ax = plt.subplots()

ax.scatter(x = train['TotalBsmtSF'], y = train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('TotalBsmtSF', fontsize=13)

plt.show()
sns.distplot(train['SalePrice'], fit = norm);
train['SalePrice'] = np.log1p(train['SalePrice'])

sns.distplot(train['SalePrice'], fit = norm);
y = train['SalePrice'].reset_index(drop = True)

train.drop(['SalePrice'], axis = 1,  inplace = True)



train.drop(['Id'], axis=1, inplace=True)

test.drop(['Id'], axis=1, inplace=True)



data = pd.concat([train, test], sort = True).reset_index(drop = True)
total = data.isnull().sum().sort_values(ascending = False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'MissingRatio'])

missing_data = missing_data[missing_data['MissingRatio'] > 0.0]

missing_data.head(10)
data.drop(['PoolQC',  'MiscFeature', 'Alley',  'Fence', 'FireplaceQu'], axis=1, inplace = True)
data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))

data['MSZoning'] = data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))



attributes = ['Electrical', 'Functional', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType', 'Utilities']

for attribute in attributes:

    data[attribute] = data[attribute].fillna(data[attribute].mode()[0])

# For categorical 

for col in ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'MasVnrType']:

    data[col] = data[col].fillna('None')

# For numerical 

for col in ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',

            'GarageYrBlt', 'GarageArea', 'GarageCars', 'MasVnrArea']:

    data[col] = data[col].fillna(0)
total = data.isnull().sum().sort_values(ascending = False)

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

missing_data = pd.concat([total, percent], axis = 1, keys = ['Total', 'MissingRatio'])

missing_data = missing_data[missing_data['MissingRatio'] > 0.0]



missing_data
numerics = data.dtypes[data.dtypes != "object"].index



data.loc[data['TotalBsmtSF'] > 0, 'TotalBsmtSF'] = boxcox1p(data['TotalBsmtSF'], 0.15)



skewness = data[numerics].apply(lambda x: skew(x))

skew_index = skewness[abs(skewness) > 0.5].index

skewness = skewness[skew_index].sort_values(ascending = False)

for idx in skew_index:

    data[idx] = boxcox1p(data[idx], 0.15)
data['hasGarage'] = data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

data['hasBsmt'] = data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

data['hasPool'] = data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

data['hasFireplace'] = data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

data['has2ndfloor'] = data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
data = pd.get_dummies(data).reset_index(drop=True)



features = data.keys()

data = data.drop(data.loc[:, (data == 0).sum() >= (data.shape[0] * 0.99)], axis = 1)

data = data.drop(data.loc[:, (data == 1).sum() >= (data.shape[0] * 0.99)], axis = 1)

remove = [feat for feat in features if feat not in data.keys()]



print("Del %2d features"%(len(remove)))
data = pd.DataFrame(RobustScaler().fit_transform(data))
data_train = np.array(data[:len(train)])

data_test = np.array(data[len(train):])



kfolds = KFold(n_splits = 10, shuffle = True, random_state = 42)

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



def cv_rmse(model, x = data_train):

    rmse = np.sqrt(-cross_val_score(model, x, y, scoring="neg_mean_squared_error", cv=kfolds))

    return (rmse)



alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]

e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]



ridge = make_pipeline(RobustScaler(), RidgeCV(alphas = alphas_alt, cv = kfolds))

lasso = make_pipeline(RobustScaler(), LassoCV(max_iter = 1e7, alphas = alphas2, random_state = 42, cv = kfolds))

elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter = 1e7, alphas = e_alphas, cv = kfolds, l1_ratio = e_l1ratio))                                

svr = make_pipeline(RobustScaler(), SVR(C = 20, epsilon = 0.008, gamma = 0.0003))



gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, 

        min_samples_split=10, loss='huber', random_state =42)

xgb = XGBRegressor(learning_rate=0.01,n_estimators=3460, max_depth=3, min_child_weight=0, gamma=0, subsample=0.7, 

        colsample_bytree=0.7, objective ='reg:squarederror', nthread=-1, scale_pos_weight=1, seed=27, reg_alpha=0.00006)

lgb = LGBMRegressor(objective='regression', num_leaves=4, learning_rate=0.01, n_estimators=5000, max_bin=200, 

        bagging_fraction=0.75, bagging_freq=5, bagging_seed=7, feature_fraction=0.2, feature_fraction_seed=7, verbose=-1)

                

stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, gbr, xgb, lgb), meta_regressor = xgb, use_features_in_secondary = True)



stack_gen_model = stack_gen.fit(data_train, y)

elastic_model = elasticnet.fit(data_train, y)

lasso_model = lasso.fit(data_train, y)

ridge_model = ridge.fit(data_train, y)

svr_model = svr.fit(data_train, y)

gbr_model = gbr.fit(data_train, y)

xgb_model = xgb.fit(data_train, y)

lgb_model = lgb.fit(data_train, y)
def models_predict(x):

    return ((0.1 * ridge_model.predict(x)) + \

            (0.05 * lasso_model.predict(x)) + \

            (0.1 * elastic_model.predict(x)) + \

            (0.1 * svr_model.predict(x)) + \

            (0.1 * gbr_model.predict(x)) + \

            (0.15 * xgb_model.predict(x)) + \

            (0.1 * lgb_model.predict(x)) + \

            (0.3 * stack_gen_model.predict(x)))



pre =  np.floor(np.expm1(models_predict(data_test)))

submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

submission.iloc[:,1] = pre

submission.to_csv("submission.csv", index=False)