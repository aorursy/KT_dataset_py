# libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# load data

train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



# numerical and categorical features

numerical = train.select_dtypes(include=['int64','float64'])

categorical = train.select_dtypes(include=['object'])



print('Train shape:', train.shape)

print('Test shape:', test.shape)
# correlation matrix

f, ax = plt.subplots(figsize=(12, 9))

corrmat = train.corr()

sns.heatmap(corrmat, vmax=.8, square=True)
# sale price top 10 highest correlations matrix

cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 9}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# visualizing missing data

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data.head(20))



# drop columns with above 15% missing

# garage features expressed by GarageCars

# basement features expressed by TotalBsmtSF

# mas features are not essential

# keep electrical, remove 1 missing row



# dropping columns

train = train.drop((missing_data[missing_data['Total'] > 1]).index, axis=1)

test = test.drop((missing_data[missing_data['Total'] > 1]).index, axis=1)   # whatever columns we drop in training need to be dropped in test set too

train = train.drop(['Utilities'], axis=1)                                   # 1 "NoSeWa", 2 NA, rest "Allpub". 'NoSewa' is in train set, won't help in prediction. drop it.

test = test.drop(['Utilities'], axis=1)  

train = train.drop(train.loc[train['Electrical'].isnull()].index)

print('Train Missing Features:', train.isnull().sum().max()) # double check



# handling missing data in test set now

total = test.isnull().sum().sort_values(ascending=False)

percent = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

print(missing_data.head(20))



test["Functional"] = test["Functional"].fillna("Typ")                               # data description says NA = 'typical'

for col in ('MSZoning', 'Exterior1st', 'Exterior2nd', 'KitchenQual', 'SaleType'):

    test[col] = test[col].fillna(test[col].mode()[0])

for col in ('BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageCars', 'GarageArea', 'TotalBsmtSF'):

    test[col] = test[col].fillna(0)



print('Test Missing Features:', test.isnull().sum().max()) # double check



# Afternote: looking back, I should have concatenated the train and test set to preprocess the data together. Lesson learned.
# outliers



from sklearn.preprocessing import StandardScaler

from scipy import stats



# data normalization/standardization (mean of 0, std of 1)

saleprice_scaled = StandardScaler().fit_transform(train['SalePrice'][:,np.newaxis])       # newaxis = None, adds new column

lows = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

highs = saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('Lows:')

print(lows)

print('\nHighs:')

print(highs)
# saleprice vs grlivarea

var = 'GrLivArea'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))



# note the two lower outliers. we will delete these. 

# the two higher outliers can stay since they go with the trend



# deleting points

train.sort_values(by = 'GrLivArea', ascending = False)[:2]

train = train.drop(train[train['Id'] == 1299].index)

train = train.drop(train[train['Id'] == 524].index)
# saleprice vs totalbsmtSF

var = 'TotalBsmtSF'

data = pd.concat([train['SalePrice'], train[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))



# deleting points with TotalBsmtSF > 3000

train = train.drop(train[train[var] > 3000].index)
# sale price normalization/standardization



from scipy.stats import norm

fig, ax = plt.subplots(2, 2, figsize=(10, 8))



# plot sale price

sns.distplot(train['SalePrice'], fit=norm, ax=ax[0, 0])

res = stats.probplot(train['SalePrice'], plot=ax[0, 1])

print("Skewness:", train['SalePrice'].skew())

print("Kurtosis:", train['SalePrice'].kurt())



# apply log transformation

train['SalePrice'] = np.log(train['SalePrice'])



# transformed sale price plot

sns.distplot(train['SalePrice'], fit=norm, ax=ax[1, 0])

res = stats.probplot(train['SalePrice'], plot=ax[1, 1])

print("Transformed Skewness:", train['SalePrice'].skew())

print("Transformed Kurtosis:", train['SalePrice'].kurt())
# GrLiveArea normalization/standardization



fig, ax = plt.subplots(2, 2, figsize=(10, 8))



# plot grlivarea

sns.distplot(train['GrLivArea'], fit=norm, ax=ax[0, 0])

res = stats.probplot(train['GrLivArea'], plot=ax[0, 1])

print("Skewness:", train['GrLivArea'].skew())

print("Kurtosis:", train['GrLivArea'].kurt())



# apply log transformation

train['GrLivArea'] = np.log(train['GrLivArea'])

test['GrLivArea'] = np.log(test['GrLivArea'])



# transformed grlivarea plot

sns.distplot(train['GrLivArea'], fit=norm, ax=ax[1, 0])

res = stats.probplot(train['GrLivArea'], plot=ax[1, 1])

print("Transformed Skewness:", train['GrLivArea'].skew())

print("Transformed Kurtosis:", train['GrLivArea'].kurt())
# TotalBsmtSF normalization/standardization



fig, ax = plt.subplots(2, 2, figsize=(10, 8))



# plot TotalBsmtSF

sns.distplot(train['TotalBsmtSF'], fit=norm, ax=ax[0, 0])

res = stats.probplot(train['TotalBsmtSF'], plot=ax[0, 1])

print("Skewness:", train['TotalBsmtSF'].skew())

print("Kurtosis:", train['TotalBsmtSF'].kurt())



# TotalBsmtSF has many 0 values, which means we can't do log transformations

# create a binary variable of HasBasement 

# take log of all non-zero observations

train['HasBsmt'] = pd.Series(0, index=train.index)

test['HasBsmt'] = pd.Series(0, index=test.index)

train.loc[train['TotalBsmtSF'] > 0,'HasBsmt'] = 1

test.loc[test['TotalBsmtSF'] > 0,'HasBsmt'] = 1

train.loc[train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(train['TotalBsmtSF'])

test.loc[test['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(test['TotalBsmtSF'])



# transformed TotalBsmtSF price plot

sns.distplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], fit=norm, ax=ax[1, 0])

res = stats.probplot(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=ax[1, 1])

print("Transformed Skewness:", train['TotalBsmtSF'].skew())

print("Transformed Kurtosis:", train['TotalBsmtSF'].kurt())
# checking for homoscedasticity - equal levels of variance across the range 

# use scatterplots



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(train['GrLivArea'], train['SalePrice'])

ax1.set_title('SalePrice vs GrLivArea')

ax2.scatter(train[train['TotalBsmtSF'] > 0]['TotalBsmtSF'], train[train['TotalBsmtSF'] > 0]['SalePrice'])

ax2.set_title('SalePrice vs TotalBsmtSF')
# convert categorical variables into dummy codes



y_train = train.SalePrice.values

X_train = train.drop(['SalePrice'], axis=1)



nTrain = X_train.shape[0]

data = pd.concat((X_train, test)).reset_index(drop=True)

data = pd.get_dummies(data)

print(data.shape)



X_train = data[:nTrain]

test = data[nTrain:]
""" MODELS """



# libraries

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

from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization
# shuffled cross validation



n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)  

    rmse = np.sqrt( - cross_val_score(model, X_train.values, y_train, scoring="neg_mean_squared_error", cv=kf))

    return rmse.mean()



def r2_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)  

    r2 = cross_val_score(model, X_train.values, y_train, scoring="r2", cv=kf)

    return r2.mean()
# LASSO regression



def lasso_target(alpha):

    lasso = make_pipeline(RobustScaler(), Lasso(alpha=alpha, random_state=1))

    score = r2_cv(lasso)

    return score



params = {'alpha': (1e-5,1e-3)}



lasso_bo = BayesianOptimization(f=lasso_target, pbounds=params, random_state=1)

lasso_bo.maximize(n_iter=25, init_points=10)

params = lasso_bo.max['params']

print(params)



lasso = make_pipeline(RobustScaler(), Lasso(alpha=params['alpha'], random_state=1))
# elastic net regression



def enet_target(alpha, l1_ratio):

    ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=1))

    score = r2_cv(ENet)

    return score



params = {'alpha': (1e-5,1e-3), 'l1_ratio': (0, 1)}



enet_bo = BayesianOptimization(f=enet_target, pbounds=params, random_state=1)

enet_bo.maximize(n_iter=25, init_points=10)

params = enet_bo.max['params']

print(params)



ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'], random_state=3))
# kernel ridge regression



def KRR_target(alpha, degree, coef0):

    KRR = KernelRidge(alpha=alpha, kernel='linear', degree=degree, coef0=coef0)

    score = r2_cv(KRR)

    return score



params = {'alpha': (0, 1), 'degree': (0, 5), 'coef0': (0, 5)}



KRR_bo = BayesianOptimization(f=KRR_target, pbounds=params, random_state=1)

KRR_bo.maximize(n_iter=25, init_points=10)

params = KRR_bo.max['params']

print(params)



KRR = KernelRidge(alpha=params['alpha'], kernel='linear', degree=params['degree'], coef0=params['coef0'])
# gradient boosting regression



def GBoost_target(n_estimators, learning_rate, max_depth, min_samples_leaf, min_samples_split):

    GBoost = GradientBoostingRegressor(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth), max_features='sqrt',

                                       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, loss='huber', random_state=5)

    score = r2_cv(GBoost)

    return score



params = {'learning_rate': (1e-4, 5e-2), 'min_samples_split': (1e-5, 0.5), 

          'min_samples_leaf': (1e-5, 0.5), 'max_depth': (3, 5), 'n_estimators': (50, 3500)}



GBoost_bo = BayesianOptimization(f=GBoost_target, pbounds=params, random_state=1)

GBoost_bo.maximize(n_iter=25, init_points=10)

params = GBoost_bo.max['params']

print(params)



GBoost = GradientBoostingRegressor(n_estimators=int(params['n_estimators']), learning_rate=params['learning_rate'], max_depth=int(params['max_depth']), max_features='sqrt',

                             min_samples_leaf=params['min_samples_leaf'], min_samples_split=params['min_samples_split'], loss='huber', random_state=5)
# XGBoost regression



def xgb_target(colsample_bytree, gamma, learning_rate, max_depth, min_child_weight, n_estimators, reg_alpha, reg_lambda, subsample):

    model_xgb = xgb.XGBRegressor(colsample_bytree=colsample_bytree, gamma=gamma, learning_rate=learning_rate, max_depth=int(max_depth), 

                                 min_child_weight=min_child_weight, n_estimators=int(n_estimators), reg_alpha=reg_alpha, reg_lambda=reg_lambda,

                                 subsample=subsample, silent=1, random_state=7, nthread=-1)

    score = r2_cv(model_xgb)

    return score



params = {'colsample_bytree': (0, 1), 'gamma': (0, 1), 'learning_rate': (1e-4, 5e-2), 

          'max_depth': (3, 10), 'min_child_weight': (1, 5), 'n_estimators': (50, 3000),

          'reg_alpha': (0, 1), 'reg_lambda': (0, 1), 'subsample': (0, 1)}



xgb_bo = BayesianOptimization(f=xgb_target, pbounds=params, random_state=1)

xgb_bo.maximize(n_iter=25, init_points=10)

params = xgb_bo.max['params']

print(params)



model_xgb = xgb.XGBRegressor(colsample_bytree=params['colsample_bytree'], gamma=params['gamma'], learning_rate=params['learning_rate'], max_depth=int(params['max_depth']), 

                             min_child_weight=params['min_child_weight'], n_estimators=int(params['n_estimators']), reg_alpha=params['reg_alpha'], reg_lambda=params['reg_lambda'],

                             subsample=params['subsample'], silent=1, random_state=7, nthread=-1)
# LightGBM



def lgb_target(num_leaves, learning_rate, n_estimators, max_bin, bagging_fraction, bagging_freq, feature_fraction, min_data_in_leaf, min_sum_hessian_in_leaf):

    model_lgb = lgb.LGBMRegressor(num_leaves=int(num_leaves), learning_rate=learning_rate, n_estimators=int(n_estimators),

                                  max_bin=int(max_bin), bagging_fraction=bagging_fraction, bagging_freq=int(bagging_freq), feature_fraction=feature_fraction,

                                  feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=int(min_data_in_leaf), min_sum_hessian_in_leaf=min_sum_hessian_in_leaf)

    score = r2_cv(model_lgb)

    return score



params = {'num_leaves': (5, 40), 'learning_rate': (1e-4, 1), 'n_estimators': (50, 3000), 'max_bin': (2, 500), 'bagging_fraction': (0, 1), 

          'bagging_freq': (1, 10), 'feature_fraction': (0, 1), 'min_data_in_leaf': (1, 40), 'min_sum_hessian_in_leaf': (0, 1)}



lgb_bo = BayesianOptimization(f=lgb_target, pbounds=params, random_state=1)

lgb_bo.maximize(n_iter=25, init_points=10)

params = lgb_bo.max['params']

print(params)



model_lgb = lgb.LGBMRegressor(num_leaves=int(params['num_leaves']), learning_rate=params['learning_rate'], n_estimators=int(params['n_estimators']),

                              max_bin=int(params['max_bin']), bagging_fraction=params['bagging_fraction'], bagging_freq=int(params['bagging_freq']), 

                              feature_fraction=params['feature_fraction'], feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf=int(params['min_data_in_leaf']), 

                              min_sum_hessian_in_leaf=params['min_sum_hessian_in_leaf'])
# RMSE scores

score = rmsle_cv(lasso)

print("\nLasso score: {:.4f}\n".format(score))

score = rmsle_cv(ENet)

print("ElasticNet score: {:.4f}\n".format(score))

score = rmsle_cv(KRR)

print("Kernel Ridge score: {:.4f}\n".format(score))

score = rmsle_cv(GBoost)

print("Gradient Boosting score: {:.4f}\n".format(score))

score = rmsle_cv(model_xgb)

print("Xgboost score: {:.4f}\n".format(score))

score = rmsle_cv(model_lgb)

print("LGBM score: {:.4f}\n".format(score))
# stacking and averaging models



class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, models):

        self.models = models

        

    # define clones of original models to fit data in

    def fit(self, X, y):

        self.models_ = [clone(x) for x in self.models]

        

        # Train cloned base models

        for model in self.models_:

            model.fit(X, y)



        return self

    

    # perform predictions for cloned models and average them

    def predict(self, X):

        predictions = np.column_stack([

            model.predict(X) for model in self.models_

        ])

        return np.mean(predictions, axis=1)   

    

averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)

print(" Averaged base models score: {:.4f}\n".format(score))
# stacking with a meta model



class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):

    def __init__(self, base_models, meta_model, n_folds=5):

        self.base_models = base_models

        self.meta_model = meta_model

        self.n_folds = n_folds

   

    # Fit data on clones of original models

    def fit(self, X, y):

        self.base_models_ = [list() for x in self.base_models]

        self.meta_model_ = clone(self.meta_model)

        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)

        

        # Train cloned base models, then create out-of-fold predictions needed to train the cloned meta-model

        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))

        for i, model in enumerate(self.base_models):

            for train_index, holdout_index in kfold.split(X, y):

                instance = clone(model)

                self.base_models_[i].append(instance)

                instance.fit(X[train_index], y[train_index])

                y_pred = instance.predict(X[holdout_index])

                out_of_fold_predictions[holdout_index, i] = y_pred

                

        # train cloned meta-model using out-of-fold predictions as features

        self.meta_model_.fit(out_of_fold_predictions, y)

        return self

   

    # perform predictions of all base models on test data 

    # use averaged predictions as meta-features for final prediction, performed by the meta-model

    def predict(self, X):

        meta_features = np.column_stack([

            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1) for base_models in self.base_models_ ])

        return self.meta_model_.predict(meta_features)

    

stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR), meta_model = lasso)

score = rmsle_cv(stacked_averaged_models)

print("Stacking Averaged models score: {:.4f}".format(score))
# ensemble of stacked regressors, XGBoost, and LightGBM



# define rmsle evaluation

def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))



stacked_averaged_models.fit(X_train.values, y_train)

stacked_train_pred = stacked_averaged_models.predict(X_train.values)

stacked_pred = np.expm1(stacked_averaged_models.predict(test.values))

print('Stacked Regressors:', rmsle(y_train, stacked_train_pred))



model_xgb.fit(X_train, y_train)

xgb_train_pred = model_xgb.predict(X_train)

xgb_pred = np.expm1(model_xgb.predict(test))

print('XGBoost:', rmsle(y_train, xgb_train_pred))



model_lgb.fit(X_train, y_train)

lgb_train_pred = model_lgb.predict(X_train)

lgb_pred = np.expm1(model_lgb.predict(test.values))

print('LightGBM:', rmsle(y_train, lgb_train_pred))
# testing different weights for ensemble



print('0.10, 0.15, 0.75:', rmsle(y_train, stacked_train_pred*0.10 + xgb_train_pred*0.15 + lgb_train_pred*0.75))

print('0.10, 0.10, 0.80:', rmsle(y_train, stacked_train_pred*0.10 + xgb_train_pred*0.10 + lgb_train_pred*0.80))   # smallest error out of ensembles

print('0.10, 0.20, 0.70:', rmsle(y_train, stacked_train_pred*0.10 + xgb_train_pred*0.20 + lgb_train_pred*0.70))

print('0.15, 0.25, 0.60:', rmsle(y_train, stacked_train_pred*0.15 + xgb_train_pred*0.25 + lgb_train_pred*0.60))

print('LightGBM:', rmsle(y_train, lgb_train_pred))                                                                # lowest train error, yet submission = 0.13, worse than ensemble
ensemble = stacked_pred*0.10 + xgb_pred*0.10 + lgb_pred*0.80



sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['SalePrice'] = ensemble

sub.to_csv('submission.csv',index=False)