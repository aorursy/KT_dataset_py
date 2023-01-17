import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import lightgbm as lgb

import xgboost as xgb

from catboost import CatBoostRegressor, CatBoostClassifier

import matplotlib.pyplot as plt

import time

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
plt.style.available

plt.style.use('ggplot')
%config InlineBackend.figure_format = 'svg'

%matplotlib inline
def read_data():

    '''read csv data to memory

    '''

    train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

    print('train has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')

    print('test has {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    return train, test





def run_lgb():

    '''train a lightgbm model 

    '''

    y = train['target'].values

    x = train.drop(['id', 'target'], axis = 1)

    x_test = test.drop('id', axis = 1)





    x_train, x_val, y_train, y_val = train_test_split(x, y, 

                                                      test_size = 0.2, 

                                                      random_state = 42, 

                                                      stratify = y)

    categorical_features = [c for c, col in enumerate(train.columns) if 'cat' in col]

    train_data = lgb.Dataset(x_train, label = y_train, categorical_feature = categorical_features)

    test_data = lgb.Dataset(x_val, label = y_val, categorical_feature = categorical_features)

    

    # to record eval results for plotting

    evals_result = {} 



    parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 0

     }

    start = time.time()

    model = lgb.train(parameters,

                       train_data,

                       valid_sets = [train_data, test_data],

                       num_boost_round = 500,

                       evals_result = evals_result,

                       early_stopping_rounds = 100, 

                       verbose_eval = 25)

    print('model training costs %.2f seconds'%(time.time() - start))

    

    ax = lgb.plot_metric(evals_result, metric = 'auc')

    plt.title('LightGBM Learning Curve')

    plt.show()

    

    y_pred_val = model.predict(x_val.values, num_iteration = model.best_iteration)

    val_roc = metrics.roc_auc_score(y_val, y_pred_val)

    print('Out of folds auc roc score is {:.4f}'.format(val_roc))



    y = model.predict(x_test.values, num_iteration = model.best_iteration)  

    output = pd.DataFrame({'id': test['id'], 'target': y})

    output.to_csv('submission_lgb.csv', index = False)

    return output





train, test = read_data()

output_lgb = run_lgb()

output_lgb
def run_cat():

    '''train a catboost

    '''

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

        

    y = train0['target'].values

    x = train0.drop(['id', 'target'], axis = 1)

    

    

    x_train, x_val, y_train, y_val = train_test_split(x, y, 

                                                      test_size = 0.2, 

                                                      random_state = 42, 

                                                      stratify = y)

    

    model = CatBoostClassifier(iterations = 500,

                              learning_rate = 0.05,

                              depth = 10,

                              eval_metric = 'AUC',

                              random_seed = 42,

                              bagging_temperature = 0.2,

                              od_type = 'Iter',

                              metric_period = 50,

                              od_wait = 20)

    

    start = time.time()

    model.fit(x_train, y_train,

                 eval_set = (x_val, y_val),

                 use_best_model = True,

                 cat_features = categorical_features,

                 verbose = 50,

                 plot = True)

    print('model training costs %.2f seconds'%(time.time() - start))

    y_pred_val = model.predict_proba(x_val)

    val_roc = metrics.roc_auc_score(y_val, y_pred_val[:,1])

    print('Out of folds auc roc score is {:.4f}'.format(val_roc))

    

    y = model.predict_proba(x_test0)  

    output = pd.DataFrame({'id': test['id'], 'target': y[:,1]})

    output.to_csv('submission_cat.csv', index = False)

    return output



output_cat = run_cat()

output_cat


def encoder_data(train, onehot = True):

    '''convert categorical feature to one-hot encoding

    '''

    if onehot:

        categorical_features = [col for col in train.columns if 'cat' in col]

        categorical_features.remove('ps_car_11_cat')

        feature = train[categorical_features]

        label_encoder = LabelEncoder()



        for col in categorical_features:

            train[col]= label_encoder.fit_transform(train[col]) 

    #print('categorical features:',categorical_features)

    

        onehotencoder = OneHotEncoder()

        for col in categorical_features:

            X = onehotencoder.fit_transform(train[col].values.reshape(-1,1)).toarray()

            col_name = [col + str(int(i)) for i in range(X.shape[1])]

            dfOneHot = pd.DataFrame(X, columns = col_name) 

            train.drop(col, axis=1 ,inplace =True) 

            train = pd.concat([train, dfOneHot], axis=1)

    else:

        pass

    print('Our data has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    return train
def run_xgb():

    '''train a xgboost model

    '''

    y = train['target'].values

    x = train.drop(['id', 'target'], axis = 1)

    x_test = test.drop('id', axis = 1)

    x = np.array(x)

    y = np.array(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, 

                                                      test_size = 0.2, 

                                                      random_state = 42, 

                                                      stratify = y)

    parameters = {

    'objective': 'binary:logistic',

    'eval_metric': 'auc',

    #'is_unbalance': 'true',

    'learning_rate': 0.05,

    #'verbose': 0,

    #'max_depth': 5,

    #'gamma':10,

    #'min_child_weight': 10,

    #'reg_alpha': 8,

    #'reg_lambda': 1.3

    }   

    train_data = xgb.DMatrix(x_train, label = y_train)

    test_data = xgb.DMatrix(x_val, label = y_val)

    eval_sets = [(train_data, 'train'), (test_data, 'eval')]

    evals_result = {} 

    start = time.time()

    model = xgb.train(parameters,

                       train_data,

                       evals = eval_sets,

                       num_boost_round = 500,

                       evals_result = evals_result,

                       early_stopping_rounds = 100, 

                       verbose_eval = 25)

    print('model training costs %.2f seconds'%(time.time() - start))

    y_pred_val = model.predict(xgb.DMatrix(x_val))

    val_roc = metrics.roc_auc_score(y_val, y_pred_val)

    print('Out of folds auc roc score is {:.4f}'.format(val_roc))

    

    ax = lgb.plot_metric(evals_result, metric = 'auc')

    plt.title('Xgboost Learning Curve')

    plt.show()

    

    y = model.predict(xgb.DMatrix(x_test.values)) 

    output = pd.DataFrame({'id': test['id'], 'target': y})

    output.to_csv('submission_xgb.csv', index = False)

    return output
train1 = encoder_data(train,onehot = False)

test1 = encoder_data(test,onehot = False)

output_xgb = run_xgb()

output_xgb