# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline



import xgboost as xgb

from xgboost.sklearn import XGBRegressor

from sklearn import cross_validation, metrics

from sklearn.grid_search import GridSearchCV



from matplotlib.pylab import rcParams

#rcParams['figure.figsize'] = 12, 4



from scipy.stats import skew

from scipy.stats.stats import pearsonr

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# The error metric: RMSE on the log of the sale prices.

from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):

    return np.sqrt(mean_squared_error(y_true, y_pred))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('train_modified.csv')

test = pd.read_csv('test_modified.csv')

label_df = pd.read_csv('label_df.csv')



train.shape, test.shape,label_df.shape
def modelfit(alg, dtrain, dtest,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):

    

    if useTrainCV:

        xgb_param = alg.get_xgb_params()

        xgtrain = xgb.DMatrix(dtrain, label_df)

        xgtest = xgb.DMatrix(dtest)

        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,

             early_stopping_rounds=early_stopping_rounds)

        alg.set_params(n_estimators=cvresult.shape[0])

    

    #Fit the algorithm on the data

    alg.fit(dtrain, label_df)

    

    # Run prediction on training set to get a rough idea of how well it does.

    y_pred = alg.predict(dtrain)

    y_test = label_df

    print("XGBoost score on training set: ", rmse(y_test, y_pred))
xgb1 = xgb.XGBRegressor(

                 colsample_bytree=0.2,

                 gamma=0.0,

                 learning_rate=0.1,

                 max_depth=4,

                 min_child_weight=1.5,

                 n_estimators=7200,                                                                  

                 reg_alpha=0.9,

                 reg_lambda=0.6,

                 subsample=0.2,

                 seed=42,

                 silent=1)

modelfit(xgb1, train, test)
#Grid seach on subsample and max_features

#Choose all predictors          

param_test1 = {

    'max_depth':list(range(3,10)),

    'min_child_weight':list(range(1,6,2))

}

gsearch1 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=7200, max_depth=5,

                        min_child_weight=1, gamma=0, subsample=0.2, colsample_bytree=0.2,

                        nthread=4, scale_pos_weight=1, seed=42), 

                        param_grid = param_test1,n_jobs=4,iid=False, cv=5)

gsearch1.fit(train,label_df)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
#Grid seach on subsample and max_features

#Choose all predictors          

param_test2 = {

    'max_depth':list(range(3,10)),

    'min_child_weight':list(range(1,6))

}

gsearch2 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=7200, max_depth=5,

                        min_child_weight=1, gamma=0, subsample=0.2, colsample_bytree=0.2,

                        nthread=4, scale_pos_weight=1, seed=42), 

                        param_grid = param_test2,n_jobs=4,iid=False, cv=5)

gsearch2.fit(train,label_df)
gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_
#Grid seach on subsample and max_features

#Choose all predictors          

param_test2b = {

    'max_depth':list(range(3,10)),

    'min_child_weight':list(range(1,6))

}

gsearch2b = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=7200, max_depth=5,

                        min_child_weight=1, gamma=0, subsample=0.2, colsample_bytree=0.2,

                        nthread=4, scale_pos_weight=1, seed=42), 

                        param_grid = param_test2b,n_jobs=4,iid=False, cv=5)

gsearch2b.fit(train,label_df)
gsearch2b.grid_scores_, gsearch2b.best_params_, gsearch2b.best_score_
#Grid seach on subsample and max_features

#Choose all predictors 

param_test3 = {

    'gamma':[i/10.0 for i in range(0,5)]

}

gsearch3 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=7200, max_depth=4,

                        min_child_weight=1.2, gamma=0, subsample=0.2, colsample_bytree=0.2,

                        nthread=4, scale_pos_weight=1, seed=42), 

                        param_grid = param_test3,n_jobs=4,iid=False, cv=5)

gsearch3.fit(train,label_df)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#Grid seach on subsample and max_features

#Choose all predictors 

param_test4 = {

    'subsample':[i/10.0 for i in range(1,10)],

    'colsample_bytree':[i/10.0 for i in range(1,10)]

}

gsearch4 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=7200, max_depth=4,

                        min_child_weight=1.2, gamma=0, subsample=0.2, colsample_bytree=0.2,

                        nthread=4, scale_pos_weight=1, seed=42), 

                        param_grid = param_test4,n_jobs=4,iid=False, cv=5)

gsearch4.fit(train,label_df)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_
#Grid seach on subsample and max_features

#Choose all predictors 

param_test6 = {

    'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]

}

gsearch6 = GridSearchCV(estimator = XGBRegressor( learning_rate =0.1, n_estimators=7200, max_depth=4,

                        min_child_weight=1.2, gamma=0, subsample=0.2, colsample_bytree=0.2,

                        nthread=4, scale_pos_weight=1, seed=42), 

                        param_grid = param_test6,n_jobs=4,iid=False, cv=5)

gsearch6.fit(train,label_df)
gsearch6.grid_scores_, gsearch6.best_params_, gsearch6.best_score_