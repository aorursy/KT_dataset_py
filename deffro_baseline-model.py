import os

import time

from math import sqrt

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor

from sklearn import linear_model

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

#pandas settings

pd.set_option('max_colwidth',250)

pd.set_option('max_columns',250)

pd.set_option('max_rows',500)
train = pd.read_csv('../input/duth-dbirlab2-1/train.csv')

test = pd.read_csv('../input/duth-dbirlab2-1/test.csv')
train.head()
for df in [train,test]:

    for c in df.drop(['obs_id'],axis=1):

        if (df[c].dtype=='object'):

            lbl = LabelEncoder() 

            lbl.fit(list(df[c].values))

            df[c] = lbl.transform(list(df[c].values))
# Some useful parameters which will come in handy later on

ntrain = train.shape[0] # or len(train)

ntest = test.shape[0] # or len(test)

SEED = 11 # for reproducibility

NFOLDS = 5 # set folds for out-of-fold prediction

folds = KFold(n_splits= NFOLDS, random_state=SEED, shuffle=True)
cols_to_exclude = ['obs_id','Overall Probability']

df_train_columns = [c for c in train.columns if c not in cols_to_exclude]



y_train = train['Overall Probability'].ravel() #ravel coverts a series to a numpy array

x_train = train[df_train_columns].values # converts a dataframe to a numpy array

x_test = test[df_train_columns].values
def train_model(X_train, X_test, Y_train, folds=5, model_type='lgb',plot_feature_importance=True):



    oof = np.zeros(ntrain)

    prediction = np.zeros(ntest)

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X_train,Y_train)):

        print('Fold', fold_n+1, 'started at', time.ctime())

        x_train, x_valid = X_train[train_index], X_train[valid_index]

        y_train, y_valid = Y_train[train_index], Y_train[valid_index] 

        

        

        

        

        if model_type == 'ridge':

            model = linear_model.Ridge(alpha=.5)

            model.fit(x_train, y_train)

            y_pred_valid = model.predict(x_valid)

            y_pred = model.predict(X_test)         

        

        if model_type == 'linear':

            model = LinearRegression()

            model.fit(x_train, y_train)

            y_pred_valid = model.predict(x_valid)

            y_pred = model.predict(X_test) 

            

        if model_type == 'rf':

            model = RandomForestRegressor(min_weight_fraction_leaf=0.05,n_jobs=-2,random_state=0, max_depth=4, n_estimators=100)

            model.fit(x_train, y_train)

            y_pred_valid = model.predict(x_valid)

            y_pred = model.predict(X_test)               

        

        if model_type == 'lgb':

            lgb_params = {   

                         'num_leaves': 7,

                         'min_data_in_leaf': 20,

                         'min_sum_hessian_in_leaf': 11,

                         'objective': 'regression',

                         'max_depth': 6,

                         'learning_rate': 0.05,

                         'boosting': "gbdt",

                         'feature_fraction': 0.8,

                         'feature_fraction_seed': 9,

                         'max_bin ': 55,

                         "bagging_freq": 5,

                         "bagging_fraction": 0.8,

                         "bagging_seed": 9,

                         'metric': 'rmse',

                         'lambda_l1': 0.1,

                         'verbosity': -1,

                         'min_child_weight': 5.34,

                         'reg_alpha': 1.130,

                         'reg_lambda': 0.360,

                         'subsample': 0.8,

                         }

            

            

            model = lgb.LGBMRegressor(**lgb_params, n_estimators = 20000, n_jobs = -1)

            model.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_valid, y_valid)], eval_metric='rmse',verbose=10000, early_stopping_rounds=100)

            

            y_pred_valid = model.predict(x_valid)

            y_pred_valid = np.clip(y_pred_valid, a_min=0, a_max=1)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            y_pred = np.clip(y_pred, a_min=0, a_max=1)

            

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = train[df_train_columns].columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)            



        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)

        prediction += y_pred          

        

    if (model_type == 'lgb' and plot_feature_importance==True):



        cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

            by="importance", ascending=False)[:50].index



        best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



        plt.figure(figsize=(16, 12));

        sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

        plt.title('LGB Features (avg over folds)')



    prediction /= NFOLDS        

    print('CV mean score: {0:.5f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    



    

    return oof, prediction
oof, prediction = train_model(X_train=x_train, X_test=x_test, Y_train=y_train, folds=folds, model_type='rf', plot_feature_importance=True)
sample_submission = pd.read_csv('../input/duth-dbirlab2-1/sample_submission.csv')

sub_df = pd.DataFrame({"obs_id":sample_submission["obs_id"].values})

sub_df["Overall Probability"] = prediction

sub_df["Overall Probability"] = sub_df["Overall Probability"].apply(lambda x: 1 if x>1 else 0 if x<0 else x)

sub_df.to_csv("submission.csv", index=False)
sub_df