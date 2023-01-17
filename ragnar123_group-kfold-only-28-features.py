import numpy as np 

import pandas as pd

import math

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

import gc

warnings.filterwarnings('ignore')

import lightgbm as lgb

from bayes_opt import BayesianOptimization

from sklearn.model_selection import GroupKFold, StratifiedKFold, train_test_split

from sklearn import metrics

from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', 10000)

pd.set_option('display.max_rows', 10000)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def read_data():

    print('Reading training, testing and submission data...')

    train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

    test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')

    submission = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time':str})

    print('Train set has {} rows and {} columns'.format(train.shape[0], train.shape[1]))

    print('Test set has {} rows and {} columns'.format(test.shape[0], test.shape[1]))

    return train, test, submission



train, test, submission = read_data()
# concatenate data

batch = 50

total_batches = 14

train['set'] = 'train'

test['set'] = 'test'

data = pd.concat([train, test])

for i in range(int(total_batches)):

    data.loc[(data['time'] > i * batch) & (data['time'] <= (i + 1) * batch), 'batch'] = i + 1

train = data[data['set'] == 'train']

test = data[data['set'] == 'test']

train.drop(['set'], inplace = True, axis = 1)

test.drop(['set'], inplace = True, axis = 1)

del data
# clean batch 8 (check signal vs time of this batch and you will see some rare event)

del_ind1 = train[(train['batch']==8) & (train['time']>=364) & (train['time']<=382) & (train['signal'] > 5)].index

del_ind2 = train[(train['batch']==8) & (train['time']>=364) & (train['time']<=382) & (train['signal'] < 0)].index

train.drop(del_ind1, axis = 0, inplace = True)

train.drop(del_ind2, axis = 0, inplace = True)

train.reset_index(drop = True, inplace = True)
def rolling_features(train, test):

    

    pre_train = train.copy()

    pre_test = test.copy()

    

        

    for df in [pre_train, pre_test]:

        for window in [1000, 5000, 10000, 20000]:

            

            # roll backwards

            df['signalstd_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).std())

            df['signalvar_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).var())

            df['signalmin_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).min())

            df['signalmax_t' + str(window)] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(1).rolling(window).max())



            min_max = (df['signal'] - df['signalmin_t' + str(window)]) / (df['signalmax_t' + str(window)] - df['signalmin_t' + str(window)])

            df['norm_t' + str(window)] = min_max * (np.floor(df['signalmax_t' + str(window)]) - np.ceil(df['signalmin_t' + str(window)]))



            # roll forward

            df['signalstd_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).std())

            df['signalvar_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).var())

            df['signalmin_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).min())

            df['signalmax_t' + str(window) + '_lead'] = df.groupby(['batch'])['signal'].transform(lambda x: x.shift(- window - 1).rolling(window).max())

            

            min_max = (df['signal'] - df['signalmin_t' + str(window) + '_lead']) / (df['signalmax_t' + str(window) + '_lead'] - df['signalmin_t' + str(window) + '_lead'])

            df['norm_t' + str(window) + '_lead'] = min_max * (np.floor(df['signalmax_t' + str(window) + '_lead']) - np.ceil(df['signalmin_t' + str(window) + '_lead']))

                

    del train, test, min_max

    

    return pre_train, pre_test



def static_batch_features(df):

    

    # thanks to https://www.kaggle.com/jazivxt/physically-possible for this feature engineering part

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df.index = ((df.time * 10_000) - 1).values

    df['batch'] = df.index // 25_000

    df['batch_index'] = df.index  - (df.batch * 25_000)

    df['batch_slices'] = df['batch_index']  // 2500

    df['batch_slices2'] = df.apply(lambda r: '_'.join([str(r['batch']).zfill(3), str(r['batch_slices']).zfill(3)]), axis=1)



    for c in ['batch','batch_slices2']:

        d = {}

        d['mean'+c] = df.groupby([c])['signal'].mean()

        d['median'+c] = df.groupby([c])['signal'].median()

        d['max'+c] = df.groupby([c])['signal'].max()

        d['min'+c] = df.groupby([c])['signal'].min()

        d['std'+c] = df.groupby([c])['signal'].std()

        d['mean_abs_chg'+c] = df.groupby([c])['signal'].apply(lambda x: np.mean(np.abs(np.diff(x))))

        d['abs_max'+c] = df.groupby([c])['signal'].apply(lambda x: np.max(np.abs(x)))

        d['abs_min'+c] = df.groupby([c])['signal'].apply(lambda x: np.min(np.abs(x)))

        d['range'+c] = d['max'+c] - d['min'+c]

        d['maxtomin'+c] = d['max'+c] / d['min'+c]

        d['abs_avg'+c] = (d['abs_min'+c] + d['abs_max'+c]) / 2

        for v in d:

            df[v] = df[c].map(d[v].to_dict())



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    df.reset_index(drop = True, inplace = True)

        

    return df



# feature engineering

pre_train1, pre_test1 = rolling_features(train, test)

pre_train2 = static_batch_features(train)

pre_test2 = static_batch_features(test)



# join features for training

feat = [col for col in pre_train2.columns if col not in ['open_channels', 'signal', 'time', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]

pre_train = pd.concat([pre_train1, pre_train2[feat]], axis = 1)

pre_test = pd.concat([pre_test1, pre_test2[feat]], axis = 1)

del pre_train1, pre_train2, pre_test1, pre_test2



del train, test

gc.collect()
def run_lgb(pre_train, pre_test, features, params, get_sample = True, bayesian = True, verbose_eval = False):

    

    pre_train = pre_train.copy()

    pre_test = pre_test.copy()

    

    # get a random sample for faster training

    if get_sample:

        pre_train = pre_train.sample(frac = 0.1, random_state = 20)

    

    pre_train.reset_index(drop = True, inplace = True)

    pre_test.reset_index(drop = True, inplace = True)

    

    # groupkfold to predict and evaluate unknown batches

    kf = GroupKFold(n_splits = 10)

    target = 'open_channels'

    oof_pred = np.zeros(len(pre_train))

    y_pred = np.zeros(len(pre_test))

     

    for fold, (tr_ind, val_ind) in enumerate(kf.split(pre_train, groups = pre_train['batch'])):

        x_train, x_val = pre_train[features].iloc[tr_ind], pre_train[features].iloc[val_ind]

        y_train, y_val = pre_train[target][tr_ind], pre_train[target][val_ind]

        train_set = lgb.Dataset(x_train, y_train)

        val_set = lgb.Dataset(x_val, y_val)

        

        model = lgb.train(params, train_set, num_boost_round = 10000, early_stopping_rounds = 50, 

                         valid_sets = [train_set, val_set], verbose_eval = verbose_eval)

        

        oof_pred[val_ind] = model.predict(x_val)

        

        y_pred += model.predict(pre_test[features]) / kf.n_splits

        

    rmse_score = np.sqrt(metrics.mean_squared_error(pre_train[target], oof_pred))

    # want to clip and then round predictions (you can get a better performance using optimization to found the best cuts)

    oof_pred = np.round(np.clip(oof_pred, 0, 10)).astype(int)

    round_y_pred = np.round(np.clip(y_pred, 0, 10)).astype(int)

    f1 = metrics.f1_score(pre_train[target], oof_pred, average = 'macro')

    

    if bayesian:

        return rmse_score

    else:

        print(f'Our oof rmse score is {rmse_score}')

        print(f'Our oof macro f1 score is {f1}')

        return round_y_pred



# features were picked with forward feature selection technique

features = ['norm_t10000', 'signalstd_t1000', 'signalvar_t1000', 'norm_t1000', 'signalstd_t1000_lead', 'signalvar_t1000_lead', 'norm_t1000_lead', 'signalstd_t5000', 'signalvar_t5000', 

            'norm_t5000', 'signalstd_t5000_lead', 'signalvar_t5000_lead', 'norm_t5000_lead', 'signalstd_t10000', 'signalvar_t10000', 'norm_t10000_lead', 'signalstd_t20000', 

            'signalvar_t20000', 'stdbatch', 'rangebatch', 'stdbatch_slices2', 'rangebatch_slices2', 'maxtominbatch_slices2', 'meanbatch_msignal', 'medianbatch_msignal', 

            'maxbatch_msignal', 'rangebatch_msignal', 'meanbatch_slices2_msignal']
def run_lgb_bayesian(num_leaves, max_depth, lambda_l1, lambda_l2, bagging_fraction, bagging_freq, colsample_bytree, learning_rate):

    

    params = {

        'boosting_type': 'gbdt',

        'metric': 'rmse',

        'objective': 'regression',

        'n_jobs': -1,

        'seed': 236,

        'num_leaves': int(num_leaves),

        'learning_rate': learning_rate,

        'max_depth': int(max_depth),

        'lambda_l1': lambda_l1,

        'lambda_l2': lambda_l2,

        'bagging_fraction': bagging_fraction,

        'bagging_freq': int(bagging_freq),

        'colsample_bytree': colsample_bytree,

        'verbose': 0

    }

    

    # use samples to make bayesian optimization faster (just for experimentation purposes, better to do it with the full training set)

    rmse_score = run_lgb(pre_train, pre_test, features, params, True, True, False)

    return -rmse_score





# run bayezian optimization with optimal features



bounds_lgb = {

    'num_leaves': (20, 300),

    'max_depth': (8, 100),

    'lambda_l1': (0, 5),

    'lambda_l2': (0, 5),

    'bagging_fraction': (0.4, 1),

    'bagging_freq': (1, 10),

    'colsample_bytree': (0.4, 1),

    'learning_rate': (0.025, 0.2)

}



lgb_bo = BayesianOptimization(run_lgb_bayesian, bounds_lgb, random_state = 236)

lgb_bo.maximize(init_points = 20, n_iter = 20, acq = 'ucb', xi = 0.0, alpha = 1e-6)



params = {

    'boosting_type': 'gbdt',

    'metric': 'rmse',

    'objective': 'regression',

    'n_jobs': -1,

    'seed': 236,

    'num_leaves': int(lgb_bo.max['params']['num_leaves']),

    'learning_rate': lgb_bo.max['params']['learning_rate'],

    'max_depth': int(lgb_bo.max['params']['max_depth']),

    'lambda_l1': lgb_bo.max['params']['lambda_l1'],

    'lambda_l2': lgb_bo.max['params']['lambda_l2'],

    'bagging_fraction': lgb_bo.max['params']['bagging_fraction'],

    'bagging_freq': int(lgb_bo.max['params']['bagging_freq']),

    'colsample_bytree': lgb_bo.max['params']['colsample_bytree']}
round_y_pred = run_lgb(pre_train, pre_test, features, params, False, False, 100)

submission['open_channels'] = round_y_pred

submission.to_csv('submission.csv', index = False)