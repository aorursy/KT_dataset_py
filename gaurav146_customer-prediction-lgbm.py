# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split

from sklearn.linear_model import  ElasticNet

from sklearn.metrics import roc_auc_score

from bayes_opt import BayesianOptimization

from skopt  import BayesSearchCV 

data = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

submit = pd.read_csv("../input/sample_submission.csv")
data.shape, test.shape
data.pop("ID_code")

data.head()
data["target"].value_counts()
data.info()
x = data.drop(labels = "target", axis =1)

y = data["target"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
test.pop("ID_code")

test.head()
from sklearn.model_selection import StratifiedKFold,RepeatedKFold

import warnings
import lightgbm as lgb

def bayes_parameter_opt_lgb(x, y, init_round=15, opt_round=25, n_folds=3, random_seed=6,n_estimators=10000, output_process=False):

    # prepare data

    train_data = lgb.Dataset(data=x, label=y, free_raw_data=False)

    # parameters

    def lgb_eval(learning_rate,num_leaves, feature_fraction, bagging_fraction, max_depth, max_bin, min_data_in_leaf,min_sum_hessian_in_leaf,subsample):

        params = {'application':'binary', 'metric':'auc'}

        params['learning_rate'] = max(min(learning_rate, 1), 0)

        params["num_leaves"] = int(round(num_leaves))

        params['feature_fraction'] = max(min(feature_fraction, 1), 0)

        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

        params['max_depth'] = int(round(max_depth))

        params['max_bin'] = int(round(max_depth))

        params['min_data_in_leaf'] = int(round(min_data_in_leaf))

        params['min_sum_hessian_in_leaf'] = min_sum_hessian_in_leaf

        params['subsample'] = max(min(subsample, 1), 0)

        

        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval =200, metrics=['auc'])

        return max(cv_result['auc-mean'])

     

    lgbBO = BayesianOptimization(lgb_eval, {'learning_rate': (0.01, 1.0),

                                            'num_leaves': (24, 80),

                                            'feature_fraction': (0.1, 0.9),

                                            'bagging_fraction': (0.8, 1),

                                            'max_depth': (5, 30),

                                            'max_bin':(20,90),

                                            'min_data_in_leaf': (20, 80),

                                            'min_sum_hessian_in_leaf':(0,100),

                                           'subsample': (0.01, 1.0)}, random_state=200)



    

    #n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.

    #init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

    

    lgbBO.maximize(init_points=init_round, n_iter=opt_round)

    

    model_auc=[]

    for model in range(len( lgbBO.res)):

        model_auc.append(lgbBO.res[model]['target'])

    

    # return best parameters

    return lgbBO.res[pd.Series(model_auc).idxmax()]['target'],lgbBO.res[pd.Series(model_auc).idxmax()]['params']



opt_params = bayes_parameter_opt_lgb(x, y, init_round=5, opt_round=10, n_folds=3, random_seed=6,n_estimators=10000)
opt_params[1]["num_leaves"] = int(round(opt_params[1]["num_leaves"]))

opt_params[1]['max_depth'] = int(round(opt_params[1]['max_depth']))

opt_params[1]['min_data_in_leaf'] = int(round(opt_params[1]['min_data_in_leaf']))

opt_params[1]['max_bin'] = int(round(opt_params[1]['max_bin']))

opt_params[1]['objective']='binary'

opt_params[1]['metric']='auc'

opt_params[1]['is_unbalance']=True

opt_params[1]['boost_from_average']=False

opt_params=opt_params[1]

opt_params
import lightgbm as lgb

d_train = lgb.Dataset(x_train, label=y_train)

params = {}

params['bagging_freq'] = 5

params['bagging_fraction'] = 1.0

params['learning_rate'] = 0.4519777231136509

params['boost_from_average'] = 'false'

params['feature_fraction'] = 0.22201637485620965

params['boosting_type'] = 'gbdt'

params['objective'] = 'binary'

params['metric'] = 'auc'

params['min_data_in_leaf'] = 80

params['min_sum_hessian_in_leaf'] = 6.3210343010469074

params['num_threads'] = 8

params['num_leaves'] = 24

params['tree_learner'] = 'serial'

params['max_depth'] = -1

params['verbosity'] = -1

clf = lgb.train(params, d_train, 200)
pred = clf.predict(x_test)

roc_auc_score(y_test, pred)
submit["target"] = abs(clf.predict(test))
submit.to_csv("submit",index=False)
submit.head(30)