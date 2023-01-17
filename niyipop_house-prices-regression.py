import torch

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats

from scipy.stats import norm, skew #for some statistics



import warnings

warnings.simplefilter('ignore')



import os

print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



print('Train data : {}'.format(train.shape))

print('Test data : {}'.format(test.shape))
train.head()
test.head()
fig, ax = plt.subplots()

ax.scatter(x = train['GrLivArea'], y = train['SalePrice'])

plt.ylabel('SalePrice')

plt.xlabel('GrLivArea')

plt.show()
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)



fig, ax = plt.subplots()

ax.scatter(train['GrLivArea'], train['SalePrice'])

plt.ylabel('SalePrice', fontsize=13)

plt.xlabel('GrLivArea', fontsize=13)

plt.show()
train_size = train.shape[0]



train_Id = train['Id']

test_Id = test['Id']



train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)





sales_price = train['SalePrice'].values



train = train.drop(columns=['SalePrice'])
all_data = pd.concat((train, test)).reset_index(drop=True)



print(all_data.shape)

all_data.head()
# Perform the sales price to make it normally distributed

n_sales_price = np.log1p(sales_price)



f, axes = plt.subplots(1, 2, figsize=(10,3))

sns.distplot(pd.Series(sales_price, name="Sales Price (Before Transformation)") , fit=norm, ax=axes[0]);

sns.distplot(pd.Series(n_sales_price, name="Sales Price (After Transformation)")  , fit=norm, ax=axes[1]);
def get_missing_data(data_df):

    null_columns = data_df.columns[data_df.isnull().any()]

    missing_data = ((data_df[null_columns].isnull().sum() / len(data_df[null_columns])) *100).sort_values(ascending=False)

    missing_data = pd.DataFrame({'Missing Ratio' :missing_data})

    return missing_data



missing_data = get_missing_data(all_data)

missing_data.head()
def replace_missing_data(data_df):

    

    # These are the fields for with the missing values that we will be replaced with...

    

    # 'None'

    cols_none   = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrType']

    for col in cols_none:

        data_df[col] = data_df[col].fillna('None')

    

    

    # 0

    cols_0      = ['GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea']

    for col in cols_0:

        data_df[col] = data_df[col].fillna(0)

    

    # Median (Group by Neighboorhood)

    cols_median = ['LotFrontage']

    for col in cols_median:

        data_df[col] = data_df.groupby('Neighborhood')[col].transform(lambda x: x.fillna(x.median()))

    

    # Mode

    cols_mode   = ['Electrical', 'MSZoning', 'Utilities', 'Functional', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual']

    for col in cols_mode:

        data_df[col] = data_df[col].fillna(data_df[col].mode()[0])

        

    return data_df



all_data = replace_missing_data(all_data)

all_data.head()
# Check to confirm that there is no missing value anymore

missing_data = get_missing_data(all_data)

print(len(missing_data))
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)

all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)



skewness = pd.DataFrame({'Skew' :skewed_feats})



print(skewness.shape)

skewness.head()
def plot_visual_skewness():

    col = 6

    row = 5

    f, axes = plt.subplots(row, col, figsize=(20,18))

    

    n = 0

    for i in range(row):

        for j in range(col):

            field = skewness.index[n]

            sns.distplot(all_data[field] , fit=norm, ax=axes[i][j]);

            n += 1

            

plot_visual_skewness()
from scipy.special import boxcox1p

skewed_features = skewness.index

lam = 0.15

for feature in skewed_features:

    all_data[feature] = boxcox1p(all_data[feature], lam)

    

plot_visual_skewness()
all_data = pd.get_dummies(all_data)

all_data.shape
x_train = all_data[:train_size]

x_test = all_data[train_size:]



y_train = n_sales_price



((x_train.shape, y_train.shape), x_test.shape)
x_train.head()
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
n_folds = 5



def rmsle_cv(model):

    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train.values)

    rmse= np.sqrt(-cross_val_score(model, x_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
# LASSO Regression

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))



# Elastic Net Regression

ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))



# Kernel Ridge Regression

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



# Gradient Boost Regression

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', 

                                   min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)



# XGBoost

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200, reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1, random_state =7, nthread = -1)



# LightGBM

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5, learning_rate=0.05, n_estimators=720,

                              max_bin = 55, bagging_fraction = 0.8, bagging_freq = 5, feature_fraction = 0.2319,

                              feature_fraction_seed=9, bagging_seed=9, min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)



models = {'LASSO': lasso, 

          'Elastic Net': ENet, 

          'Kernel Ridge Regression': KRR, 

          'Gradient Boost': GBoost, 

          'XGBoost': model_xgb, 

          'LightGBM': model_lgb}
# for key, value in models.items():

#     score = rmsle_cv(value)

#     print( key + " score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
# class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):

#     def __init__(self, models):

#         self.models = models

        

#     # we define clones of the original models to fit the data in

#     def fit(self, X, y):

#         self.models_ = [clone(x) for x in self.models]

        

#         # Train cloned base models

#         for model in self.models_:

#             model.fit(X, y)



#         return self

    

#     #Now we do the predictions for cloned models and average them

#     def predict(self, X):

#         predictions = np.column_stack([

#             model.predict(X) for model in self.models_

#         ])

#         return np.mean(predictions, axis=1)   
# averaged_models = AveragingModels(models = (ENet, GBoost, model_lgb, lasso))



# score = rmsle_cv(averaged_models)

# print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
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
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, GBoost, KRR),

                                                 meta_model = lasso)



# stacked_averaged_models = StackingAveragedModels(base_models = (model_xgb, model_lgb, KRR),

#                                                  meta_model = GBoost)



# score = rmsle_cv(stacked_averaged_models)

# print("Stacking Averaged models score: {:.4f} ({:.4f})".format(score.mean(), score.std()))
def rmsle(y, y_pred):

    return np.sqrt(mean_squared_error(y, y_pred))
models = {'LASSO': lasso, 

          'Elastic Net': ENet, 

          'Kernel Ridge Regression': KRR, 

          'Gradient Boost': GBoost, 

          'XGBoost': model_xgb, 

          'LightGBM': model_lgb,

          'StackedRegressor' : stacked_averaged_models}
# for key, model in models.items():

    

#     model.fit(x_train.values, y_train)

#     train_pred = model.predict(x_train.values)

#     print( key + ': {:.4f}'.format(rmsle(y_train, train_pred)))

    

# #     pred = np.expm1(model.predict(x_test.values))



GBoost.fit(x_train.values, y_train)

train_pred = GBoost.predict(x_train.values)

print('Gradient Boost : {:.4f}'.format(rmsle(y_train, train_pred)))
Id = test_Id

SalePrice = np.expm1(GBoost.predict(x_test.values))



my_submission = pd.DataFrame({'Id': Id, 'SalePrice': SalePrice})

my_submission.to_csv('submission.csv', index=False)



my_submission.head()
# def get_ensemble(data):

#     ensemble =  (0.7 * stacked_averaged_models.predict(data)) + (0.1 * GBoost.predict(data)) + (0.1 * model_xgb.predict(data)) + (0.1 * model_lgb.predict(data))

# #     ensemble =  (0.7 * np.expm1(stacked_averaged_models.predict(data))) + (0.1 * np.expm1(GBoost.predict(data))) + (0.1 * np.expm1(model_xgb.predict(data))) + (0.1 * np.expm1(model_lgb.predict(data)))

    

#     return ensemble



# train_ensemble = get_ensemble(x_train.values)

# print('RMSLE score on train data:')

# print(rmsle(y_train, train_ensemble))
# ensemble = np.expm1(get_ensemble(x_test.values))
# Id = test_Id

# SalePrice = ensemble



# my_submission = pd.DataFrame({'Id': Id, 'SalePrice': SalePrice})

# my_submission.to_csv('submission.csv', index=False)



# my_submission.head()
# from sklearn.linear_model import LinearRegression



# reg = LinearRegression().fit(x_train, y_train)

# reg.score(x_train, y_train)







# reg.coef_

# reg.intercept_
# y_test = reg.predict(x_test)



# Id = test_Id

# SalePrice = y_test



# my_submission = pd.DataFrame({'Id': Id, 'SalePrice': SalePrice})

# my_submission.to_csv('submission.csv', index=False)



# my_submission.head()