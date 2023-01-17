#libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.pylab import rcParams



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

from sklearn.pipeline import Pipeline



import warnings

warnings.filterwarnings("ignore") #to ignore warning for readability



with warnings.catch_warnings():

    warnings.filterwarnings("ignore",category=DeprecationWarning)



#Read the data

X = pd.read_csv("../input/iowa-house-prices/train.csv", index_col = 'Id')

X_test = pd.read_csv("../input/iowa-house-prices/test.csv", index_col ='Id')



#Filter from target column null values

X.dropna(axis = 0, subset = ['SalePrice'], inplace = True)

y = X['SalePrice']



#Filter out the target column from X dataset

X.drop(axis = 1, labels = ['SalePrice'], inplace = True)
#Verify if there are the same number of columns in both test and train data

print(X.shape)

print((X.columns == X_test.columns).sum())
X.columns
# MoSold, YrSOld, SaleType, SaleCondition won't be available for a prediction for the new house

#so these columns will be dropped

leakage_columns = ['MoSold', 'YrSold', 'SaleType', 'SaleCondition']

X.drop(labels = leakage_columns, axis = 1, inplace = True)

X_test.drop(labels = leakage_columns, axis = 1, inplace= True)
#select only columns that are int or float

numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]



#check how many nulls these have

cols_with_nulls = X[numerical_cols].isnull().sum()

print(cols_with_nulls[cols_with_nulls > 0])
sns.set_style('whitegrid')

df = X[['LotFrontage', 'MasVnrArea', 'GarageYrBlt']].dropna(axis = 0)



#LotFrontage

#we'll impute the mode, as the distributions is asymetric and the mean is influenced by outliers

plt.figure(figsize = (12,4))

sns.distplot(a = df['LotFrontage'], bins = 30, norm_hist=False, kde=True, color = 'blue')
#MasVnrArea

#we'll use most frequent, as the distribution is strongly asymetric to the right

plt.figure(figsize = (12,4))

sns.distplot(a = df['MasVnrArea'], bins = 30, norm_hist=False, kde=True, color = 'purple')
#GarageYrBlt

#for this distribution mean should work just fine :)

plt.figure(figsize = (12,4))

sns.distplot(a = df['GarageYrBlt'], bins = 30, norm_hist=False, kde=True, color = 'orange')
#There are only 3 columns with missing values, and they are small in number, so we'll apply Simple Imputation

from sklearn.impute import SimpleImputer



numerical_cols_median = ['LotFrontage']

numerical_transformer_median = SimpleImputer(strategy = 'median')



numerical_cols_mod = ['MasVnrArea']

numerical_transformer_mod = SimpleImputer(strategy = 'most_frequent')



numerical_cols_mean = ['GarageYrBlt']

numerical_transformer_mean = SimpleImputer(strategy = 'mean')



numerical_cols_remain = set(numerical_cols) - set(numerical_cols_mean) - set(numerical_cols_median) - set(numerical_cols_mod)

numerical_cols_remain = list(numerical_cols_remain)
#choosing only columns with categorical data

categorical_cols = [col for col in X.columns if X[col].dtype == 'object']



#checking their missing values

cols_with_nulls_categs = X[categorical_cols].isnull().sum()

cols_with_nulls_categs[cols_with_nulls_categs > 0]
#Because there are a few columns with too many missing values, we'll filter them out from the data

X.drop(labels = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)

X_test.drop(labels = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis = 1, inplace = True)



#Redo categorical_cols variable

categorical_cols = [col for col in X.columns if X[col].dtype == 'object']





#performing imputer and OHE to encode the features

from sklearn.preprocessing import OneHotEncoder



categ_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy = 'most_frequent')),

                                         ('onehot', OneHotEncoder(handle_unknown = 'ignore'))])
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers=[('num_median', numerical_transformer_median, numerical_cols_median),

                                               ('num_mod', numerical_transformer_mod, numerical_cols_mod),

                                               ('num_mean', numerical_transformer_mean, numerical_cols_mean),

                                               ('num_rest', numerical_transformer_mean, numerical_cols_remain),

                                              ('cat', categ_transformer, categorical_cols)])
#split the training data into train & valid

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 0)



import xgboost as xgb

from xgboost import XGBRegressor

from sklearn import model_selection, metrics

from sklearn.model_selection import GridSearchCV



rcParams['figure.figsize'] = 14, 4



def modelfit(model):

    pipeline = Pipeline(steps=[('preprocessing', preprocessor),

                              ('model', model)])

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds)

    print('MAE:', mae)

    
xgb1 = XGBRegressor( learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8,

                     colsample_bytree=0.8, nthread=4, scale_pos_weight=1, seed=27, objective='reg:squarederror')



modelfit(xgb1)
#Best Values



#dictionaire of parameters to grid search

param_test1 = { 'max_depth' : range(3, 10, 2), 'min_child_weight' : range(1, 6, 2)}

X_train_grid = preprocessor.fit_transform(X_train) #apply transformations



gsearch1 = GridSearchCV(estimator= XGBRegressor(learning_rate =0.1, n_estimators=1000, 

                                                gamma=0, subsample=0.8, colsample_bytree=0.8, 

                                                nthread=4, scale_pos_weight=1, seed=27, objective='reg:squarederror'), 

                        param_grid = param_test1, cv=5)

gsearch1.fit(X_train_grid, y_train)



#outputs

print('Best params:', gsearch1.best_params_)

print('Best estim:', gsearch1.best_estimator_)

print('Best score:', gsearch1.best_score_)
#Optimum Values

param_test2 = {'max_depth' : [3, 4, 5], 'min_child_weight' : [1]}



gsearch2 = GridSearchCV(estimator= XGBRegressor(learning_rate =0.1, n_estimators=1000, 

                                                gamma=0, subsample=0.8, colsample_bytree=0.8, 

                                                nthread=4, scale_pos_weight=1, seed=27, objective='reg:squarederror'), 

                        param_grid = param_test2, cv=5)

gsearch2.fit(X_train_grid, y_train)



print('Best params:', gsearch2.best_params_)

print('Best estim:', gsearch2.best_estimator_)

print('Best score:', gsearch2.best_score_)
param_test3 = {'gamma' : [i/10.0 for i in range(0,5)]}

gsearch3 = GridSearchCV(estimator= XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bynode=1, 

                                                colsample_bytree=0.8, gamma=0, importance_type='gain', learning_rate=0.1, 

                                                max_delta_step=0, max_depth=4, min_child_weight=1, missing=None, 

                                                n_estimators=1000, n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0, 

                                                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27, silent=None, 

                                                subsample=0.8, verbosity=1), 

                        param_grid = param_test3, cv=5)

gsearch3.fit(X_train_grid, y_train)



print('Best params:', gsearch3.best_params_)

print('Best estim:', gsearch3.best_estimator_)

print('Best score:', gsearch3.best_score_)
# first try

param_test4 = {'subsample' : [i/10.0 for i in range(6,10)], 'colsample_bytree' : [i/10.0 for i in range(6,10)]}

gsearch4 = GridSearchCV(XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                                     colsample_bynode=1, gamma=0.0,

                                     importance_type='gain', learning_rate=0.1, max_delta_step=0,

                                     max_depth=4, min_child_weight=1, missing=None, n_estimators=1000,

                                     n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

                                     reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27,

                                     silent=None, verbosity=1),

                       param_grid = param_test4, cv=5)

gsearch4.fit(X_train_grid, y_train)



print('Best params:', gsearch4.best_params_)

print('Best estim:', gsearch4.best_estimator_)

print('Best score:', gsearch4.best_score_)



# going deeper

param_test5 = {'subsample' : [i/100.0 for i in range(60,80,5)], 'colsample_bytree' : [i/100.0 for i in range(70,90,5)]}

gsearch5 = GridSearchCV(XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                                     colsample_bynode=1, gamma=0.0,

                                     importance_type='gain', learning_rate=0.1, max_delta_step=0,

                                     max_depth=4, min_child_weight=1, missing=None, n_estimators=1000,

                                     n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

                                     reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=27,

                                     silent=None, verbosity=1),

                       param_grid = param_test5, cv=5)

gsearch5.fit(X_train_grid, y_train)



print('Best params:', gsearch4.best_params_)

print('Best estim:', gsearch4.best_estimator_)

print('Best score:', gsearch4.best_score_)
# first try

param_test6 = {'reg_alpha' : [1e-5, 1e-2, 0.1, 1, 100]}

gsearch6 = GridSearchCV(XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                                     colsample_bynode=1, colsample_bytree=0.7, gamma=0.0,

                                     importance_type='gain', learning_rate=0.1, max_delta_step=0,

                                     max_depth=4, min_child_weight=1, missing=None, n_estimators=1000,

                                     n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

                                     reg_lambda=1, scale_pos_weight=1, seed=27,

                                     silent=None, subsample=0.8, verbosity=1),

                       param_grid = param_test6, cv=5)



gsearch6.fit(X_train_grid, y_train)



print('Best params:', gsearch6.best_params_)

print('Best estim:', gsearch6.best_estimator_)

print('Best score:', gsearch6.best_score_)



# going deeper :)

param_test7 = {'reg_alpha' : [0.9, 1, 1.1]}

gsearch7 = GridSearchCV(XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

                                     colsample_bynode=1, colsample_bytree=0.7, gamma=0.0,

                                     importance_type='gain', learning_rate=0.1, max_delta_step=0,

                                     max_depth=4, min_child_weight=1, missing=None, n_estimators=1000,

                                     n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

                                     reg_lambda=1, scale_pos_weight=1, seed=27,

                                     silent=None, subsample=0.8, verbosity=1),

                       param_grid = param_test7, cv=5)



gsearch7.fit(X_train_grid, y_train)



print('Best params:', gsearch7.best_params_)

print('Best estim:', gsearch7.best_estimator_)

print('Best score:', gsearch7.best_score_)
xgb3 = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.8, gamma=0.0,

             importance_type='gain', learning_rate=0.005, max_delta_step=0,

             max_depth=4, min_child_weight=1, missing=None, n_estimators=10000,

             n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

             reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=27,

             silent=None, subsample=0.8, verbosity=1)



modelfit(xgb3)
#my model with all tunings

my_model = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,

             colsample_bynode=1, colsample_bytree=0.8, gamma=0.0,

             importance_type='gain', learning_rate=0.005, max_delta_step=0,

             max_depth=4, min_child_weight=1, missing=None, n_estimators=5000,

             n_jobs=1, nthread=4, objective='reg:squarederror', random_state=0,

             reg_alpha=1, reg_lambda=1, scale_pos_weight=1, seed=27,

             silent=None, subsample=0.8, verbosity=1)



pipeline_final = Pipeline(steps=[('preprocessor', preprocessor),

                                ('model', my_model)])

pipeline_final.fit(X, y)



preds_test = pipeline_final.predict(X_test)

output = pd.DataFrame({'Id': X_test.index, 'SalePrice' : preds_test})

output.to_csv('submission_final_v3.2.csv', index = False)