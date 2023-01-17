"""
PLAsTiCC Astronomical Classification Feature Extraction
----------------------------------
@website https://www.kaggle.com/mithrillion/know-your-objective/

Goal :
------
We are starting with basic modeling technique explained in Oliver's kernal to understand implementation of lightgbms and generating features for big data.

Train 5 lightgbms on the meta_data + aggregated data

Then go through test data in chunks and generate predictions

New in this version :
---------------------
1. This versions adds some of the Flux calculations made available by MichaelApers https://www.kaggle.com/michaelapers
    here https://www.kaggle.com/michaelapers/the-plasticc-astronomy-starter-kit
2. class 99 mean adjustment

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import gc
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import logging
import os
print(os.listdir("../input"))
def create_logger():
    logger_ = logging.getLogger('main')
    logger_.setLevel(logging.DEBUG)
    fh = logging.FileHandler('simple_lightgbm.log')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)s]%(asctime)s:%(name)s:%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger_.addHandler(fh)
    logger_.addHandler(ch)
def get_logger():
    return logging.getLogger('main')
def lgb_multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order='F')

    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return 'wloss', loss, False
def multi_weighted_logloss(y_true, y_preds):
    """
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    # class_weights taken from Giba's topic : https://www.kaggle.com/titericz
    # https://www.kaggle.com/c/PLAsTiCC-2018/discussion/67194
    # with Kyle Boone's post https://www.kaggle.com/kyleboone
    classes = [6, 15, 16, 42, 52, 53, 62, 64, 65, 67, 88, 90, 92, 95]
    class_weight = {6: 1, 15: 2, 16: 1, 42: 1, 52: 1, 53: 1, 62: 1, 64: 2, 65: 1, 67: 1, 88: 1, 90: 1, 92: 1, 95: 1}
    if len(np.unique(y_true)) > 14:
        classes.append(99)
        class_weight[99] = 2
    y_p = y_preds
    # Trasform y_true in dummies
    y_ohe = pd.get_dummies(y_true)
    # Normalize rows and limit y_preds to 1e-15, 1-1e-15
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    # Transform to log
    y_p_log = np.log(y_p)
    # Get the log for ones, .values is used to drop the index of DataFrames
    # Exclude class 99 for now, since there is no class99 in the training set
    # we gave a special process for that class
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    # Get the number of positives for each class
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    # Weight average and divide by the number of positives
    class_arr = np.array([class_weight[k] for k in sorted(class_weight.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = - np.sum(y_w) / np.sum(class_arr)
    return loss
def get_aggregations():
    return {
        # Dropped mjd aggregations on CPMP advice
        # see https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696
        # 'mjd': ['min', 'max', 'size'],
        'passband': ['mean', 'std', 'var'],  # ''min', 'max', 'mean', 'median', 'std'],
        'flux': ['min', 'max', 'mean', 'median', 'std'],
        'flux_err': ['min', 'max', 'mean', 'median', 'std'],
        'detected': ['mean'],  # ''min', 'max', 'mean', 'median', 'std'],
    }


def get_new_columns(aggs):
    return [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


def add_features_to_agg(df):
    # CPMP using the following feature was really silliy :)
    # df['mjd_diff'] = df['mjd_max'] - df['mjd_min']
    # see https://www.kaggle.com/c/PLAsTiCC-2018/discussion/69696
    
    # The others may be useful
    df['flux_diff'] = df['flux_max'] - df['flux_min']
    df['flux_dif2'] = (df['flux_max'] - df['flux_min']) / df['flux_mean']
    df['flux_w_mean'] = df['flux_by_flux_ratio_sq_sum'] / df['flux_ratio_sq_sum']
    df['flux_dif3'] = (df['flux_max'] - df['flux_min']) / df['flux_w_mean']

    # del df['mjd_max'], df['mjd_min']

    return df
def Generate_Features_Chunk(df_, meta_, features, train_mean):

    df_['flux_ratio_sq'] = np.power(df_['flux'] / df_['flux_err'], 2.0)
    df_['flux_by_flux_ratio_sq'] = df_['flux'] * df_['flux_ratio_sq']

    # Group by object id
    aggs = get_aggregations()

    aggs = get_aggregations()
    aggs['flux_ratio_sq'] = ['sum']
    aggs['flux_by_flux_ratio_sq'] = ['sum']

    new_columns = get_new_columns(aggs)

    agg_ = df_.groupby('object_id').agg(aggs)
    agg_.columns = new_columns

    agg_ = add_features_to_agg(df=agg_)

    # Merge with meta data
    full_test = agg_.reset_index().merge(
        right=meta_,
        how='left',
        on='object_id'
    )

    full_test = full_test.fillna(train_mean)
    return full_test

def main():
    train = pd.read_csv('../input/training_set.csv')
    train['flux_ratio_sq'] = np.power(train['flux'] / train['flux_err'], 2.0)
    train['flux_by_flux_ratio_sq'] = train['flux'] * train['flux_ratio_sq']

    # train = pd.concat([train, pd.get_dummies(train['passband'], prefix='passband')], axis=1, sort=False)

    aggs = get_aggregations()
    aggs['flux_ratio_sq'] = ['sum']
    aggs['flux_by_flux_ratio_sq'] = ['sum']

    # passbands = [f for f in train if 'passband_' in f]
    # get_logger().info('Passband features : {}'.format(passbands))
    # for pb in passbands:
    #     aggs[pb] = ['mean']

    agg_train = train.groupby('object_id').agg(aggs)
    new_columns = get_new_columns(aggs)
    agg_train.columns = new_columns

    agg_train = add_features_to_agg(df=agg_train)
    
    agg_train.head()

    del train
    gc.collect()

    meta_train = pd.read_csv('../input/training_set_metadata.csv')
    meta_train.head()

    full_train = agg_train.reset_index().merge(
        right=meta_train,
        how='outer',
        on='object_id'
    )

    train_mean = full_train.mean(axis=0)
    full_train.fillna(train_mean, inplace=True)
    get_logger().info(full_train.columns)
    
    #create feature set for training dataset
    full_train.to_csv('Training_Features.csv', index=True, float_format='%.6f')
    
    meta_test = pd.read_csv('../input/test_set_metadata.csv')

    import time

    start = time.time()
    chunks = 5000000
    remain_df = None
    
    def the_unique(x):
        return [x[i] for i in range(len(x)) if x[i] != x[i-1]]

    for i_c, df in enumerate(pd.read_csv('../input/test_set.csv', chunksize=chunks, iterator=True)):
        # Check object_ids
        # I believe np.unique keeps the order of group_ids as they appear in the file
        # My belief is wrong (I should have read the doc !)
        # A big thank you to https://www.kaggle.com/filby89
        # Use .tolist() is almost 3 times faster than the_unique(df['object_id'].values)
        unique_ids = the_unique(df['object_id'].tolist())
        new_remain_df = df.loc[df['object_id'] == unique_ids[-1]].copy()

        if remain_df is None:
            df = df.loc[df['object_id'].isin(unique_ids[:-1])].copy()
        else:
            df = pd.concat([remain_df, df.loc[df['object_id'].isin(unique_ids[:-1])]], axis=0)

        # Create remaining samples df
        remain_df = new_remain_df

        test_features_df = Generate_Features_Chunk(df_=df,
                                 meta_=meta_test,
                                 features=full_train.columns,
                                 train_mean=train_mean)
                
        if i_c == 0:
            test_features_df.to_csv('Test_Features.csv', header=True, index=False, float_format='%.6f')
        else:
            test_features_df.to_csv('Test_Features.csv', header=False, mode='a', index=False, float_format='%.6f')

        del test_features_df
        gc.collect()

        if (i_c + 1) % 10 == 0:
            get_logger().info('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
            print('%15d done in %5.1f' % (chunks * (i_c + 1), (time.time() - start) / 60))
       # Compute last object in remain_df

    test_features_df = Generate_Features_Chunk(df_=remain_df,
                                 meta_=meta_test,
                                 features=full_train.columns,
                                 train_mean=train_mean)

    test_features_df.to_csv('Test_Features.csv', header=False, mode='a', index=False, float_format='%.6f')

if __name__ == '__main__':
    gc.enable()
    create_logger()
    try:
        main()
    except Exception:
        get_logger().exception('Unexpected Exception Occured')
        raise