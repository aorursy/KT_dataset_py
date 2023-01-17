import numpy as np 

import gc

import pandas as pd

pd.set_option("display.max_columns", 999)

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

import warnings

warnings.filterwarnings("ignore")

import seaborn as sns



from sklearn.preprocessing import StandardScaler, normalize

from sklearn.preprocessing import OneHotEncoder, LabelEncoder



from sklearn.model_selection import cross_val_score, cross_validate

from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit, train_test_split, StratifiedKFold, TimeSeriesSplit



from time import time, ctime



import catboost

from catboost import CatBoostClassifier, Pool



from sklearn import metrics

from sklearn.metrics import f1_score, roc_auc_score



RANDOM_STATE = 12061985

np.random.seed(RANDOM_STATE)



# HPO

from skopt.space import Integer, Categorical, Real

from skopt.utils import use_named_args

from skopt import gp_minimize, gbrt_minimize, forest_minimize, dummy_minimize

from skopt.plots import plot_convergence

from skopt.callbacks import DeltaXStopper, DeadlineStopper, DeltaYStopper, EarlyStopper
def get_params_SKopt(model, X, Y, space, cv_search, alg = 'catboost', cat_features = None, eval_dataset = None, UBM = False, opt_method = 'gbrt_minimize', 

                     verbose = True,  multi = False, scoring = 'neg_mean_squared_error', n_best = 50, total_time = 7200):

    

    if alg == 'catboost':

        fitparam = { 'eval_set' : eval_dataset,

                     'use_best_model' : UBM,

                     'cat_features' : cat_features,

                     'early_stopping_rounds': 10 }

    else:

        fitparam = {}

        

    @use_named_args(space)

    def objective(**params):

        model.set_params(**params)

        return -np.mean(cross_val_score(model, 

                                        X, Y, 

                                        cv=cv_search, 

                                        scoring= scoring,

                                        fit_params=fitparam))

    

    if opt_method == 'gbrt_minimize':

        

        HPO_PARAMS = {'n_calls':1000,

                      'n_random_starts':20,

                      'acq_func':'EI',}

        

        reg_gp = gbrt_minimize(objective, 

                               space, 

                               n_jobs = -1,

                               verbose = verbose,

                               callback = [DeltaYStopper(delta = 0.01, n_best = 5), RepeatedMinStopper(n_best = n_best), DeadlineStopper(total_time = total_time)],

                               **HPO_PARAMS,

                               random_state = RANDOM_STATE)

        



    elif opt_method == 'forest_minimize':

        

        HPO_PARAMS = {'n_calls':1000,

                      'n_random_starts':20,

                      'acq_func':'EI',}

        

        reg_gp = forest_minimize(objective, 

                               space, 

                               n_jobs = -1,

                               verbose = verbose,

                               callback = [RepeatedMinStopper(n_best = n_best), DeadlineStopper(total_time = total_time)],

                               **HPO_PARAMS,

                               random_state = RANDOM_STATE)

        

    elif opt_method == 'gp_minimize':

        

        HPO_PARAMS = {'n_calls':1000,

                      'n_random_starts':20,

                      'acq_func':'gp_hedge',}        

        

        reg_gp = gp_minimize(objective, 

                               space, 

                               n_jobs = -1,

                               verbose = verbose,

                               callback = [RepeatedMinStopper(n_best = n_best), DeadlineStopper(total_time = total_time)],

                               **HPO_PARAMS,

                               random_state = RANDOM_STATE)

    

    TUNED_PARAMS = {} 

    for i, item in enumerate(space):

        if multi:

            TUNED_PARAMS[item.name.split('__')[1]] = reg_gp.x[i]

        else:

            TUNED_PARAMS[item.name] = reg_gp.x[i]

    

    return [TUNED_PARAMS,reg_gp]



class RepeatedMinStopper(EarlyStopper):

    """Stop the optimization when there is no improvement in the minimum.

    Stop the optimization when there is no improvement in the minimum

    achieved function evaluation after `n_best` iterations.

    """

    def __init__(self, n_best=50):

        super(EarlyStopper, self).__init__()

        self.n_best = n_best

        self.count = 0

        self.minimum = np.finfo(np.float).max



    def _criterion(self, result):

        if result.fun < self.minimum:

            self.minimum = result.fun

            self.count = 0

        elif result.fun > self.minimum:

            self.count = 0

        else:

            self.count += 1



        return self.count >= self.n_best
train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')

test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')

submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')



train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

test = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
del train_identity, train_transaction, test_identity, test_transaction
one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]

one_value_cols_test = [col for col in test.columns if test[col].nunique() <= 1]

one_value_cols == one_value_cols_test
# From https://www.kaggle.com/artgor/eda-and-models

train['TransactionAmt_to_mean_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_mean_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('mean')

train['TransactionAmt_to_std_card1'] = train['TransactionAmt'] / train.groupby(['card1'])['TransactionAmt'].transform('std')

train['TransactionAmt_to_std_card4'] = train['TransactionAmt'] / train.groupby(['card4'])['TransactionAmt'].transform('std')



test['TransactionAmt_to_mean_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_mean_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('mean')

test['TransactionAmt_to_std_card1'] = test['TransactionAmt'] / test.groupby(['card1'])['TransactionAmt'].transform('std')

test['TransactionAmt_to_std_card4'] = test['TransactionAmt'] / test.groupby(['card4'])['TransactionAmt'].transform('std')



train['id_02_to_mean_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('mean')

train['id_02_to_mean_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('mean')

train['id_02_to_std_card1'] = train['id_02'] / train.groupby(['card1'])['id_02'].transform('std')

train['id_02_to_std_card4'] = train['id_02'] / train.groupby(['card4'])['id_02'].transform('std')



test['id_02_to_mean_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('mean')

test['id_02_to_mean_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('mean')

test['id_02_to_std_card1'] = test['id_02'] / test.groupby(['card1'])['id_02'].transform('std')

test['id_02_to_std_card4'] = test['id_02'] / test.groupby(['card4'])['id_02'].transform('std')



train['D15_to_mean_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('mean')

train['D15_to_mean_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('mean')

train['D15_to_std_card1'] = train['D15'] / train.groupby(['card1'])['D15'].transform('std')

train['D15_to_std_card4'] = train['D15'] / train.groupby(['card4'])['D15'].transform('std')



test['D15_to_mean_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('mean')

test['D15_to_mean_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('mean')

test['D15_to_std_card1'] = test['D15'] / test.groupby(['card1'])['D15'].transform('std')

test['D15_to_std_card4'] = test['D15'] / test.groupby(['card4'])['D15'].transform('std')



train['D15_to_mean_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('mean')

train['D15_to_mean_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('mean')

train['D15_to_std_addr1'] = train['D15'] / train.groupby(['addr1'])['D15'].transform('std')

train['D15_to_std_addr2'] = train['D15'] / train.groupby(['addr2'])['D15'].transform('std')



test['D15_to_mean_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('mean')

test['D15_to_mean_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('mean')

test['D15_to_std_addr1'] = test['D15'] / test.groupby(['addr1'])['D15'].transform('std')

test['D15_to_std_addr2'] = test['D15'] / test.groupby(['addr2'])['D15'].transform('std')
train[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = train['P_emaildomain'].str.split('.', expand=True)

train[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = train['R_emaildomain'].str.split('.', expand=True)

test[['P_emaildomain_1', 'P_emaildomain_2', 'P_emaildomain_3']] = test['P_emaildomain'].str.split('.', expand=True)

test[['R_emaildomain_1', 'R_emaildomain_2', 'R_emaildomain_3']] = test['R_emaildomain'].str.split('.', expand=True)
# from https://www.kaggle.com/nroman/lgb-single-model-lb-0-9419



# New feature - decimal part of the transaction amount

train['TransactionAmt_decimal'] = ((train['TransactionAmt'] - train['TransactionAmt'].astype(int)) * 1000).astype(int)

test['TransactionAmt_decimal'] = ((test['TransactionAmt'] - test['TransactionAmt'].astype(int)) * 1000).astype(int)



# Count encoding for card1 feature. 

# Explained in this kernel: https://www.kaggle.com/nroman/eda-for-cis-fraud-detection

train['card1_count_full'] = train['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))

test['card1_count_full'] = test['card1'].map(pd.concat([train['card1'], test['card1']], ignore_index=True).value_counts(dropna=False))



# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature

train['Transaction_day_of_week'] = np.floor((train['TransactionDT'] / (3600 * 24) - 1) % 7)

test['Transaction_day_of_week'] = np.floor((test['TransactionDT'] / (3600 * 24) - 1) % 7)

train['Transaction_hour'] = np.floor(train['TransactionDT'] / 3600) % 24

test['Transaction_hour'] = np.floor(test['TransactionDT'] / 3600) % 24



# Some arbitrary features interaction

for feature in ['id_02__id_20', 'id_02__D8', 'D11__DeviceInfo', 'DeviceInfo__P_emaildomain', 'P_emaildomain__C2', 

                'card2__dist1', 'card1__card5', 'card2__id_20', 'card5__P_emaildomain', 'addr1__card1']:



    f1, f2 = feature.split('__')

    train[feature] = train[f1].astype(str) + '_' + train[f2].astype(str)

    test[feature] = test[f1].astype(str) + '_' + test[f2].astype(str)

    

for feature in ['id_34', 'id_36']:

        # Count encoded for both train and test

        train[feature + '_count_full'] = train[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        test[feature + '_count_full'] = test[feature].map(pd.concat([train[feature], test[feature]], ignore_index=True).value_counts(dropna=False))

        

for feature in ['id_01', 'id_31', 'id_33', 'id_35', 'id_36']:

        # Count encoded separately for train and test

        train[feature + '_count_dist'] = train[feature].map(train[feature].value_counts(dropna=False))

        test[feature + '_count_dist'] = test[feature].map(test[feature].value_counts(dropna=False))
many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]

many_null_cols_test = [col for col in test.columns if test[col].isnull().sum() / test.shape[0] > 0.9]



big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]

big_top_value_cols_test = [col for col in test.columns if test[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]



cols_to_drop = list(set(many_null_cols + many_null_cols_test + big_top_value_cols + big_top_value_cols_test + one_value_cols+ one_value_cols_test))

cols_to_drop.remove('isFraud')

print(len(cols_to_drop))



train = train.drop(cols_to_drop, axis=1)

test = test.drop(cols_to_drop, axis=1)
cat_cols = []

for col in tqdm_notebook(train.columns):

    if train[col].dtype == 'object':

        cat_cols.append(col)

        le = LabelEncoder()

        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))

        train[col] = le.transform(list(train[col].astype(str).values))

        test[col] = le.transform(list(test[col].astype(str).values))   
X = train.sort_values('TransactionDT').drop(['isFraud', 'TransactionDT', 'TransactionID'], axis=1)

y = train.sort_values('TransactionDT')['isFraud']

test = test.sort_values('TransactionDT').drop(['TransactionDT', 'TransactionID'], axis=1)
del train

gc.collect()
# by https://www.kaggle.com/dimartinot

def clean_inf_nan(df):

    return df.replace([np.inf, -np.inf], np.nan)   



# Cleaning infinite values to NaN

X = clean_inf_nan(X)

test = clean_inf_nan(test )
X.fillna(0, inplace=True)

test.fillna(0, inplace=True)
print(X.shape, test.shape)
%%time



acc, auc, F1 = [], [], []



Y_TEST_preds = pd.DataFrame({'ind': list(test.index), 

                         'prediction': [0.0] * len(test)}) 



X_t = X.sample(frac = 1., random_state = RANDOM_STATE)

y_t = y.loc[X_t.index]



# The commented code below you can use to perform the catboost parameters tuning 

# STATIC_PARAMS = {'eval_metric': 'AUC',

#                  'random_seed': RANDOM_STATE,

#                  'scale_pos_weight': y_t.value_counts()[0]/y_t.value_counts()[1],

#                  'thread_count': -1,

#                   'task_type' : "GPU",

#                   'devices' : '0',

#                 }       



# space_SKopt = [

#          Real(0, 10, name='l2_leaf_reg'),  

#          Real(0, 10, name='bagging_temperature'),

#          Integer(1, 16, name='depth'),

#          Real(0.001, .5, name='learning_rate'),





# X_train, X_test, y_train, y_test = train_test_split(X_t, y_t, test_size=0.25, random_state=42, shuffle = True)



# eval_dataset = Pool(data=X_test,

#                 label=y_test,

#                 cat_features=cat_cols)



# n_fold = 3

# cv_tune = StratifiedKFold(n_splits=n_fold, random_state=RANDOM_STATE, shuffle=True)



# start_time = time()

# [TUNED_PARAMS,reg_gp] = get_params_SKopt(CatBoostClassifier(**STATIC_PARAMS, 

#                                                          iterations = 500,

#                                                          verbose = False,     

#                                                          gpu_cat_features_storage = 'CpuPinnedMemory',

#                                                          border_count = 10,  # 128

#                                                          max_ctr_complexity = 1, # 4

#                                                          cat_features=cat_cols,), 

#                                                          X_train, y_train, 

#                                                          space_SKopt, 

#                                                          cv_tune,

#                                                          alg = 'catboost',

#                                                          cat_features = cat_cols,

#                                                          eval_dataset = eval_dataset,

#                                                          UBM = True,

#                                                          opt_method = 'forest_minimize',

#                                                          verbose = True,

#                                                          multi = False, 

#                                                          scoring = 'roc_auc',

#                                                          n_best = 10,

#                                                          total_time = 15000)



# print('\nTime for tuning: {0:.2f} minutes'.format((time() - start_time)/60))

# NEW_PARAMS = {**STATIC_PARAMS, **TUNED_PARAMS}

# best_model = CatBoostClassifier(**NEW_PARAMS)    

# best_model.set_params(iterations = 10000,  learning_rate = TUNED_PARAMS['learning_rate']/10)

# print(best_model.get_params())



PARAMS = {'learning_rate': 0.03805602919544498,

          'depth': 4,

          'l2_leaf_reg': 2.2556764893424943,

          'random_seed': 12061985,

          'eval_metric': 'AUC',

          'bagging_temperature': 0.22312590985798633,

#           'task_type': 'GPU', 

#           'devices': '0',

          'scale_pos_weight': 27.579586700866283,

          'iterations': 10000}



best_model = CatBoostClassifier(**PARAMS)  



n_fold = 3

cv = StratifiedKFold(n_splits=n_fold, random_state=RANDOM_STATE, shuffle=True)



for fold_n, (train_index, valid_index) in enumerate(cv.split(X_t, y_t)):

    print('\nFold', fold_n, 'started at', ctime())



    X_train = X_t.iloc[train_index,:]

    X_valid = X_t.iloc[valid_index,:]



    y_train = y_t.iloc[train_index]

    y_valid = y_t.iloc[valid_index]      

    

    train_dataset = Pool(data=X_train,

                     label=y_train,

                     cat_features=cat_cols)

    

    eval_dataset = Pool(data=X_valid,

                    label=y_valid,

                    cat_features=cat_cols)

    

    best_model.fit(train_dataset,

              use_best_model=True,

              verbose = False,

              plot = True,

              eval_set=eval_dataset,

              early_stopping_rounds=50)

    

    y_pred = best_model.predict(Pool(data=X_valid, cat_features=cat_cols))



    acc.append(metrics.accuracy_score(y_valid, y_pred))

    auc.append(metrics.roc_auc_score(y_valid, y_pred))

    F1.append(metrics.f1_score(y_valid, y_pred))



    print('Best score', best_model.best_score_) 

    print('Best iteration', best_model.best_iteration_)  



    Y_TEST_preds.loc[:, 'prediction'] += best_model.predict_proba(Pool(data=test, cat_features=cat_cols))[:,1]



Y_TEST_preds.loc[:, 'prediction'] /= n_fold            



print('='*45)           

print('CV mean accuarcy: {0:.4f}, std: {1:.4f}.'.format(np.mean(acc), np.std(acc)))

print('CV mean AUC: {0:.4f}, std: {1:.4f}.'.format(np.mean(auc), np.std(auc)))

print('CV mean F1: {0:.4f}, std: {1:.4f}.'.format(np.mean(F1), np.std(F1)))
submission['isFraud'] = Y_TEST_preds['prediction'] 

submission.to_csv('cat_skopt.csv', index=False)

print(sub.head())