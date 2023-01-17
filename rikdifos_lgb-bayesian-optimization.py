%matplotlib inline

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import lightgbm as lgb

import matplotlib.pyplot as plt

import time

from bayes_opt import BayesianOptimization

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score
def read_data():

    '''read csv data to memory

    '''

    train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

    print('train has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')

    print('test has {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    return train, test





def transform_dt(train, test):

    x_test = test.drop('id', axis = 1)

    train0 = train.fillna('nan')

    x_test0 = x_test.fillna('nan')

    categorical_features = [col for col in train.columns if 'cat' in col]

    categorical_features.remove('ps_car_11_cat')

    #feature = train0[categorical_features]

    label_encoder = LabelEncoder()



    for col in categorical_features:

        train0[col]= label_encoder.fit_transform(train0[col]) 

        x_test0[col]= label_encoder.fit_transform(x_test0[col]) 

    return train0, x_test0





train, test = read_data()

train0, x_test = transform_dt(train, test)

y_train = train0['target']

x_train = train0.drop(['id', 'target'], axis = 1)



x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 

                                                  test_size = 0.2, 

                                                  random_state = 42, 

                                                  stratify = y_train)
x_train
def lgb_cv(learning_rate, num_leaves, feature_fraction, bagging_fraction):

    '''define an evaluator

    '''

    

    cat = [i for i in x_train.columns if 'cat' in i]

    

    train_set = lgb.Dataset(x_train, 

                            y_train, 

                            categorical_feature = cat)

    

    params = {'objective': 'binary',

              'metric': 'auc',

              'is_unbalance': 'true',

              'boosting': 'gbdt',

              'num_boost_round': 1500,

              'early_stopping_rounds' : 100,

              'bagging_freq': 20,

              'nthread': 4

             }

    

    params['learning_rate'] = learning_rate

    params['num_leaves'] = int(num_leaves)

    params['feature_fraction'] = max(min(feature_fraction, 1), 0)

    params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)

    

    N_folds = 5

    random_seed = 1234

    

    cv_result = lgb.cv(params, 

                       train_set, 

                       nfold=N_folds, 

                       seed=random_seed, 

                       stratified=True, 

                       verbose_eval=30, 

                       metrics=['auc'])

    

    return max(cv_result['auc-mean'])
start = time.time()

lgb_bayes_eval = BayesianOptimization(lgb_cv, {'learning_rate': (0.02, 0.1),

                                               'num_leaves': (24, 45),

                                               'feature_fraction': (0.1, 0.9),

                                               'bagging_fraction': (0.3, 0.7)

                                              })



lgb_bayes_eval.maximize(init_points=5, n_iter=10)

print('Bayesian Optimization costs %.2f seconds'%(time.time() - start))
#lgb_bayes_eval.res
best_params = max(lgb_bayes_eval.res, key=lambda x:x['target'])['params'] # get best parameters

best_params
#all_params = [x['params'] for x in lgb_bayes_eval.res]

#all_params
def validate_lgb(params2 = {'num_leaves': 31,

                            'feature_fraction': 0.5,

                            'bagging_fraction': 0.5,

                            'learning_rate': 0.05

                           } ):

    '''validate how much auc score lift after optimization

       default value: before optimization

    '''



    params1 = {'objective': 'binary',

               'metric': 'auc',

               'is_unbalance': 'true',

               'boosting': 'gbdt',

               'num_boost_round': 1500,

               'bagging_freq': 20,

               'nthread': 4

             }

    

    params1.update(params2) # add parameters after optimization



    cat = [i for i in x_train.columns if 'cat' in i]

    train_data = lgb.Dataset(x_train, 

                             label = y_train, 

                             categorical_feature = cat)

    

    val_data = lgb.Dataset(x_val, 

                           label = y_val, 

                           categorical_feature = cat)

    

    start = time.time()

    model = lgb.train(params1,

                      train_data,

                      early_stopping_rounds = 100,

                      valid_sets = [train_data, val_data], 

                      verbose_eval = 30)

    

    print('model training costs %.2f seconds'%(time.time() - start))

    

    y_pred_val = model.predict(x_val.values, num_iteration = model.best_iteration)

    val_roc = roc_auc_score(y_val, y_pred_val)

    print('Out of folds auc roc score is {:.4f}'.format(val_roc))

    return model
model = validate_lgb()
best_params['num_leaves'] = int(best_params['num_leaves']) # convert num_leaves to int

model = validate_lgb(best_params)

y_pred = model.predict(x_test.values, num_iteration = model.best_iteration)  

output = pd.DataFrame({'id': test['id'], 'target': y_pred})

output.to_csv('submission_lgb.csv', index = False)