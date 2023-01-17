import numpy as np 

import pandas as pd 

import os

import json

from pandas.io.json import json_normalize

import ast

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('ggplot')

import seaborn as sns

%matplotlib notebook

from scipy.stats import skew, boxcox

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from mpl_toolkits.mplot3d import Axes3D

import ast

import re

import yaml

import json

from collections import Counter

from nltk.corpus import stopwords

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split, KFold

import xgboost as xgb

import lightgbm as lgb

from bayes_opt import BayesianOptimization

from sklearn.metrics import mean_squared_error

from sklearn import model_selection

from sklearn.metrics import accuracy_score

import eli5

import time

from datetime import datetime

from sklearn.preprocessing import LabelEncoder

import warnings  

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)

print(os.listdir("../input"))
%%time 



train_new= pd.read_csv('../input/train-new/train_new.csv')

test_new = pd.read_csv('../input/train-new/test_new.csv')

sam_sub = pd.read_csv('../input/tmdb-box-office-prediction/sample_submission.csv')

print( "train dataset:", train_new.shape,"\n","test dataset: ",test_new.shape,"\n","sample_submission dataset:", sam_sub .shape)
#append train and test and replace NAs with mean of each column



train_new_agg=train_new

test_new_agg=test_new



train_new_agg['dataset']='train'

test_new_agg['dataset']='test'



big_and_beautiful_data=train_new_agg.append(test_new_agg)



del train_new_agg

del test_new_agg



#numerical mean replacement 

numerical_cols_df =[c for c in big_and_beautiful_data.columns if big_and_beautiful_data[c].dtype in [np.float, np.int] and c not in ['revenue', 'id']]

categorical_cols_df = [c for c in big_and_beautiful_data.columns if big_and_beautiful_data[c].dtype in [np.object]]

big_and_beautiful_data_num=big_and_beautiful_data[numerical_cols_df].fillna(big_and_beautiful_data[numerical_cols_df].mean()) 

big_and_beautiful_data= pd.concat([big_and_beautiful_data_num, big_and_beautiful_data[categorical_cols_df],big_and_beautiful_data[['revenue', 'id']]], axis=1)

#replace by mean of train and test 

numerical_cols_train =[c for c in train_new.columns if train_new[c].dtype in [np.float, np.int] and c not in ['revenue', 'id']]

categorical_cols_train = [c for c in train_new.columns if train_new[c].dtype in [np.object]]

train_new_num=train_new[numerical_cols_train].fillna(train_new[numerical_cols_train].mean()) 

train_new= pd.concat([train_new_num, train_new[categorical_cols_train],train_new[['revenue', 'id']]], axis=1)



numerical_cols_test =[c for c in test_new.columns if test_new[c].dtype in [np.float, np.int] and c not in ['id']]

categorical_cols_test = [c for c in test_new.columns if test_new[c].dtype in [np.object]]

test_new_num=test_new[numerical_cols_test].fillna(test_new[numerical_cols_test].mean()) 

test_new= pd.concat([test_new_num, test_new[categorical_cols_test],test_new[['id']]], axis=1)
#dropping unnecessary columns 

drop_columns=['homepage','imdb_id','poster_path','status','title', 'tagline', 'overview', 'original_title','all_genres','all_cast',

             'original_language','collection_name','all_crew']

train_new=train_new.drop(drop_columns,axis=1)

test_new=test_new.drop(drop_columns,axis=1)
big_and_beautiful_data=big_and_beautiful_data.drop(drop_columns,axis=1)



train_mean_agg=big_and_beautiful_data.loc[big_and_beautiful_data['dataset']=='train']

test_mean_agg=big_and_beautiful_data.loc[big_and_beautiful_data['dataset']=='test']



train_mean_agg=train_mean_agg.drop('dataset',axis=1)

test_mean_agg=test_mean_agg.drop(['dataset','revenue'],axis=1)

train_new=train_new.drop('dataset',axis=1)

test_new=test_new.drop('dataset',axis=1)





print( "updated train dataset:", train_new.shape,"\n","updated test dataset: ",test_new.shape)

print( "updated train agg dataset:", train_mean_agg.shape,"\n","updated test agg dataset: ",test_mean_agg.shape)





# Just double checking the difference of variables between train and test 

print(train_new.columns.difference(test_new.columns)) # good to go! 

print(train_mean_agg.columns.difference(test_mean_agg.columns)) # good to go! 
%%time 



X = train_new.drop(['id', 'revenue','release_date'], axis=1)

y = np.log1p(train_new['revenue'])

X_test = test_new.drop(['id','release_date'], axis=1)



dtrain = xgb.DMatrix(X, label=y)

dtest = xgb.DMatrix(X_test)



def xgb_evaluate(max_depth, subsample, eta, min_child_weight,colsample_bytree):

    params = {'eval_metric': 'rmse',

              'max_depth': int(max_depth),

              'subsample':subsample,

              'eta': eta,

              'min_child_weight': int(min_child_weight),

              'colsample_bytree': colsample_bytree}



    cv_result = xgb.cv(params, dtrain, num_boost_round=5000, nfold=3, early_stopping_rounds=50)    

    

    return -1.0 * cv_result['test-rmse-mean'].iloc[-1] #because Bayesian Optimization function maximizes, we have to flip the number by multiplying by -1







xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (0, 6), 

                                             'subsample': (0.6, 1),

                                             'eta':(0.01, 0.4),

                                             'min_child_weight':(1, 30),

                                             'colsample_bytree':(0.6, 1)})



xgb_bo.maximize(init_points=10, n_iter=15, acq='ei')





model_rmse=[]

for model in range(len(xgb_bo.res)):

    model_rmse.append(xgb_bo.res[model]['target'])

    

xgb_bo.res[pd.Series(model_rmse).idxmax()]['target']

xgb_opt_params = xgb_bo.res[pd.Series(model_rmse).idxmax()]['params']

xgb_opt_params['max_depth']= int(round(xgb_opt_params['max_depth']))

xgb_opt_params['objective']='reg:linear'

xgb_opt_params['eval_metric']='rmse'

    

xgb_opt_params 
%%time 



X = train_new.drop(['id', 'revenue','release_date'], axis=1)

y = np.log1p(train_new['revenue'])



train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)

   



def lgb_eval(learning_rate, num_leaves, bagging_fraction, max_depth, min_data_in_leaf, lambda_l1, lambda_l2):

    

    params = {

            'application': 'regression',

            'metric': 'rmse',

            'learning_rate': learning_rate,

            'num_leaves':int(round(num_leaves)),

            'bagging_fraction':bagging_fraction,   

            'max_depth': int(round(max_depth)),

            'min_data_in_leaf':int(round(min_data_in_leaf)),

            'lambda_l1':lambda_l1,

            'lambda_l2':lambda_l2}

    

    

    cv_result = lgb.cv(params, train_data, num_boost_round=5000, nfold=3,stratified =False,early_stopping_rounds=50)    

    

    return -1.0 * min(cv_result['rmse-mean'])



#put min/max of hyperparameter that you want to test 

lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 0.2),

                                            'num_leaves': (2,31 ),

                                            'bagging_fraction': (0.8, 1),

                                            'max_depth': (-1, 5),

                                            'min_data_in_leaf': (2, 20),

                                           'lambda_l1': (0, 5),

                                           'lambda_l2': (0, 5)}, random_state=200)

    

lgbBO.maximize(init_points=10, n_iter=15, acq='ei')

model_rmse=[]

for model in range(len(lgbBO.res)):

    model_rmse.append(lgbBO.res[model]['target'])

    

lgbBO.res[pd.Series(model_rmse).idxmax()]['target']

lgb_opt_params = lgbBO.res[pd.Series(model_rmse).idxmax()]['params']

lgb_opt_params['max_depth']= int(round(lgb_opt_params['max_depth']))

lgb_opt_params['num_leaves']= int(round(lgb_opt_params['num_leaves']))

lgb_opt_params['min_data_in_leaf']= int(round(lgb_opt_params['min_data_in_leaf']))



lgb_opt_params['application']='regression'

lgb_opt_params['metric']='rmse'

    
lgb_opt_params
n_fold = 10

random_seed=2222

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)





X = train_new.drop(['id', 'revenue','release_date'], axis=1)

y = np.log1p(train_new['revenue'])

X_test = test_new.drop(['id','release_date'], axis=1)

def train_model(X, X_test, y, params=None, folds=folds, model_type='lgb', plot_feature_importance=True, model=None):



    oof = np.zeros(X.shape[0])

    prediction = np.zeros(X_test.shape[0])

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        if model_type == 'sklearn':

            X_train, X_valid = X[train_index], X[valid_index]

        else:

            X_train, X_valid = X.values[train_index], X.values[valid_index]

        y_train, y_valid = y[train_index], y[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = 10000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',

                    verbose=100, early_stopping_rounds=100)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=10000, evals=watchlist, early_stopping_rounds=100, verbose_eval=100, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test.values), ntree_limit=model.best_ntree_limit)



        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_squared_error(y_valid, y_pred_valid)

            

            y_pred = model.predict(X_test)

            

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=10000,  eval_metric='RMSE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

        

        prediction += y_pred    

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction

        return oof, prediction

    

    else:

        return oof, prediction
%%time 

oof_xgb, prediction_xgb = train_model(X, X_test, y, params=xgb_opt_params, model_type='xgb')
%%time 



oof_lgb, prediction_lgb = train_model(X, X_test, y, params=lgb_opt_params, model_type='lgb')
%%time 



X = train_mean_agg.drop(['id', 'revenue','release_date'], axis=1)

y = np.log1p(train_mean_agg['revenue'])

X_test = test_mean_agg.drop(['id','release_date'], axis=1)



dtrain = xgb.DMatrix(X, label=y)

dtest = xgb.DMatrix(X_test)



def xgb_evaluate(max_depth, subsample, eta, min_child_weight,colsample_bytree):

    params = {'eval_metric': 'rmse',

              'max_depth': int(max_depth),

              'subsample':subsample,

              'eta': eta,

              'min_child_weight': int(min_child_weight),

              'colsample_bytree': colsample_bytree}



    cv_result = xgb.cv(params, dtrain, num_boost_round=5000, nfold=3, early_stopping_rounds=50)    

    

    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]







xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (0, 6), 

                                             'subsample': (0.6, 1),

                                             'eta':(0.01, 0.4),

                                             'min_child_weight':(1, 30),

                                             'colsample_bytree':(0.6, 1)})



xgb_bo.maximize(init_points=10, n_iter=15, acq='ei')



model_rmse=[]

for model in range(len(xgb_bo.res)):

    model_rmse.append(xgb_bo.res[model]['target'])

    

xgb_bo.res[pd.Series(model_rmse).idxmax()]['target']

xgb_opt_params = xgb_bo.res[pd.Series(model_rmse).idxmax()]['params']

xgb_opt_params['max_depth']= int(round(xgb_opt_params['max_depth']))

xgb_opt_params['objective']='reg:linear'

xgb_opt_params['eval_metric']='rmse'

    

xgb_opt_params
%%time 



X = train_mean_agg.drop(['id', 'revenue','release_date'], axis=1)

y = np.log1p(train_mean_agg['revenue'])

X_test = test_mean_agg.drop(['id','release_date'], axis=1)



train_data = lgb.Dataset(data=X, label=y, free_raw_data=False)

   



def lgb_eval(learning_rate, num_leaves, bagging_fraction, max_depth, min_data_in_leaf, lambda_l1, lambda_l2):

    

    params = {

            'application': 'regression',

            'metric': 'rmse',

            'learning_rate': learning_rate,

            'num_leaves':int(round(num_leaves)),

            'bagging_fraction':bagging_fraction,   

            'max_depth': int(round(max_depth)),

            'min_data_in_leaf':int(round(min_data_in_leaf)),

            'lambda_l1':lambda_l1,

            'lambda_l2':lambda_l2}

    

    

    cv_result = lgb.cv(params, train_data, num_boost_round=5000, nfold=3,stratified =False,early_stopping_rounds=50)    

    

    return -1.0 * min(cv_result['rmse-mean'])



#put min/max of hyperparameter that you want to test 

lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 0.2),

                                            'num_leaves': (2,31 ),

                                            'bagging_fraction': (0.8, 1),

                                            'max_depth': (-1, 5),

                                            'min_data_in_leaf': (2, 20),

                                           'lambda_l1': (0, 5),

                                           'lambda_l2': (0, 5)}, random_state=200)

    

    

    

lgbBO.maximize(init_points=10, n_iter=15, acq='ei')

model_rmse=[]

for model in range(len(lgbBO.res)):

    model_rmse.append(lgbBO.res[model]['target'])

    

lgbBO.res[pd.Series(model_rmse).idxmax()]['target']

lgb_opt_params = lgbBO.res[pd.Series(model_rmse).idxmax()]['params']

lgb_opt_params['max_depth']= int(round(lgb_opt_params['max_depth']))

lgb_opt_params['num_leaves']= int(round(lgb_opt_params['num_leaves']))

lgb_opt_params['min_data_in_leaf']= int(round(lgb_opt_params['min_data_in_leaf']))



lgb_opt_params['application']='regression'

lgb_opt_params['metric']='rmse'

    

lgb_opt_params
%%time 

oof_xgb_agg, prediction_xgb_agg = train_model(X, X_test, y, params=xgb_opt_params, model_type='xgb')
%%time 



oof_lgb_agg, prediction_lgb_agg = train_model(X, X_test, y, params=lgb_opt_params, model_type='lgb')
sam_sub['revenue'] = np.expm1(prediction_lgb)

sam_sub.to_csv("lgb.csv", index=False)

sam_sub['revenue'] = np.expm1(prediction_xgb)

sam_sub.to_csv("xgb.csv", index=False)



sam_sub['revenue'] = np.expm1(prediction_lgb_agg)

sam_sub.to_csv("lgb_agg.csv", index=False)

sam_sub['revenue'] = np.expm1(prediction_xgb_agg)

sam_sub.to_csv("xgb_agg.csv", index=False)







sam_sub['revenue'] = np.expm1((prediction_lgb + prediction_xgb) / 2)

sam_sub.to_csv("blend_lgb_xgb.csv", index=False)



sam_sub['revenue'] = np.expm1((prediction_lgb_agg + prediction_xgb_agg) / 2)

sam_sub.to_csv("blend_lgb_xgb_agg.csv", index=False)



sam_sub['revenue'] = np.expm1(( prediction_lgb + prediction_xgb +prediction_lgb_agg + prediction_xgb_agg) / 4)

sam_sub.to_csv("put_them_all.csv", index=False)