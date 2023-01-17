import gc

import os

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings('ignore')
def get_logger():

    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'

    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger('main')

    logger.setLevel(logging.DEBUG)

    return logger

    

logger = get_logger()
def read_data(nrows=None):

    logger.info('Input data')

    train_df = pd.read_csv('../input/santander-customer-transaction-prediction/train.csv',nrows=nrows)

    test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

    return train_df, test_df
def process_data(train_df, test_df):

    logger.info('Features engineering')

    

    synthetic = np.load('../input/publicprivate/synthetic_samples_indexes.npy')

    synthetic = synthetic-200000

    synthetic = np.array(synthetic)

    test_df = test_df.iloc[~test_df.index.isin(synthetic)]



    idx = [c for c in train_df.columns if c not in ['ID_code', 'target']]

    

    traintest = pd.concat([train_df, test_df])

    traintest = traintest.reset_index(drop=True)

    

    for col in idx:

        #find unique values

        varname = col + '_IsUnique'

        traintest[varname] = 0

        _, index_, count_ = np.unique(traintest.loc[:,col].values, return_counts=True, return_index=True)

        traintest[varname][index_[count_ == 1]] += 1

        traintest[varname] = traintest[varname] / (traintest[varname] == 1).sum()

    

    #frequency encoding

    for col in idx:

        traintest[col+'_freq'] = traintest[col].map(traintest.groupby(col).size())

    

    #Fill values from traintest to IsUnique or NotUnique. Replace zeroes with NANs

    for col in idx:

        varname = col + '_IsUnique'

        tmp_col = traintest.loc[traintest[varname] > 0][col]

        traintest[col + '_OnlyUnique'] = tmp_col

        traintest[col + '_OnlyUnique'] = traintest[col + '_OnlyUnique'].fillna(0)

        traintest[col + '_NotUnique'] = traintest[col] - traintest[col + '_OnlyUnique']

        traintest[col + '_NotUnique'] = traintest[col + '_NotUnique'].replace(0,np.nan)

        traintest[col + '_OnlyUnique'] = traintest[col + '_OnlyUnique'].replace(0,np.nan)

        traintest.pop(varname)



    train_df = traintest[:200000]

    test_df = traintest[200000:]

    

    print('Train and test shape:',train_df.shape, test_df.shape)

    return train_df, test_df

#Almost no hyperparameter tuning needed

def run_model(train_df, test_df):

    logger.info('Prepare the model')

    features = [c for c in train_df.columns if c not in ['ID_code', 'target']]

    target = train_df['target']

    logger.info('Run model')

    param = {

        'bagging_freq': 5,

        'bagging_fraction': 0.38, 

        'boost_from_average':'false',

        'boost': 'gbdt',

        'feature_fraction': 0.045,

        'learning_rate': 0.0095,

        'max_depth': -1,  #-1

        'metric':'auc',

        'min_data_in_leaf': 20,

        'min_sum_hessian_in_leaf': 10.0, 

        'num_leaves': 3,

        'num_threads': 8,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1

    }

    num_round = 1000000

    folds = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)

    oof = np.zeros(len(train_df))

    predictions = np.zeros(len(test_df))

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):

        print("Fold {}".format(fold_))

        trn_data = lgb.Dataset(train_df.iloc[trn_idx][features], label=target.iloc[trn_idx])

        val_data = lgb.Dataset(train_df.iloc[val_idx][features], label=target.iloc[val_idx])

        clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 3500)

        oof[val_idx] = clf.predict(train_df.iloc[val_idx][features], num_iteration=clf.best_iteration)

        predictions += clf.predict(test_df[features], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.5f}".format(roc_auc_score(target, oof)))

    return predictions
def submit(test_df, predictions):

    logger.info('Prepare submission')

    

    all_test_df = pd.read_csv('../input/santander-customer-transaction-prediction/test.csv')

    sub = pd.DataFrame({"ID_code": all_test_df.ID_code.values})

    sub["target"] = 0

    sub_real = pd.DataFrame({"ID_code": test_df.ID_code.values})

    sub_real["target"] = predictions

    sub = sub.set_index('ID_code')

    sub_real = sub_real.set_index('ID_code')

    sub.update(sub_real)

    sub = sub.reset_index()

    

    sub.to_csv("submission.csv", index=False)
def main(nrows=None):

    train_df, test_df = read_data(nrows)

    train_df, test_df = process_data(train_df, test_df)

    predictions = run_model(train_df, test_df)

    submit(test_df, predictions)
if __name__ == "__main__":

    main()