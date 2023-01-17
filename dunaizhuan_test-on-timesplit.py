# From kernel https://www.kaggle.com/mpearmain/extended-timeseriessplitter

"""

This module provides a class to split time-series data for back-testing and evaluation.

The aim was to extend the current sklearn implementation and extend it's uses.



Might be useful for some ;)

"""



import logging

from typing import Optional



import numpy as np

from sklearn.model_selection._split import _BaseKFold

from sklearn.utils import indexable

from sklearn.utils.validation import _num_samples



LOGGER = logging.getLogger(__name__)





class TimeSeriesSplit_(_BaseKFold):  # pylint: disable=abstract-method

    """Time Series cross-validator



    Provides train/test indices to split time series data samples that are observed at fixed time intervals,

    in train/test sets. In each split, test indices must be higher than before, and thus shuffling in cross validator is

    inappropriate.



    This cross_validation object is a variation of :class:`TimeSeriesSplit` from the popular scikit-learn package.

    It extends its base functionality to allow for expanding windows, and rolling windows with configurable train and

    test sizes and delays between each. i.e. train on weeks 1-8, skip week 9, predict week 10-11.



    In this implementation we specifically force the test size to be equal across all splits.



    Expanding Window:



            Idx / Time  0..............................................n

            1           |  train  | delay |  test  |                   |

            2           |       train     | delay  |  test  |          |

            ...         |                                              |

            last        |            train            | delay |  test  |



    Rolling Windows:

            Idx / Time  0..............................................n

            1           | train   | delay |  test  |                   |

            2           | step |  train  | delay |  test  |            |

            ...         |                                              |

            last        | step | ... | step |  train  | delay |  test  |



    Parameters:

        n_splits : int, default=5

            Number of splits. Must be at least 4.



        train_size : int, optional

            Size for a single training set.



        test_size : int, optional, must be positive

            Size of a single testing set



        delay : int, default=0, must be positive

            Number of index shifts to make between train and test sets

            e.g,

            delay=0

                TRAIN: [0 1 2 3] TEST: [4]

            delay=1

                TRAIN: [0 1 2 3] TEST: [5]

            delay=2

                TRAIN: [0 1 2 3] TEST: [6]



        force_step_size : int, optional

            Ignore split logic and force the training data to shift by the step size forward for n_splits

            e.g

            TRAIN: [ 0  1  2  3] TEST: [4]

            TRAIN: [ 0  1  2  3  4] TEST: [5]

            TRAIN: [ 0  1  2  3  4  5] TEST: [6]

            TRAIN: [ 0  1  2  3  4  5  6] TEST: [7]



    Examples

    --------

    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])

    >>> y = np.array([1, 2, 3, 4, 5, 6])

    >>> tscv = TimeSeriesSplit(n_splits=5)

    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE

    TimeSeriesSplit(train_size=None, n_splits=5)

    >>> for train_index, test_index in tscv.split(X):

    ...    print('TRAIN:', train_index, 'TEST:', test_index)

    ...    X_train, X_test = X[train_index], X[test_index]

    ...    y_train, y_test = y[train_index], y[test_index]

    TRAIN: [0] TEST: [1]

    TRAIN: [0 1] TEST: [2]

    TRAIN: [0 1 2] TEST: [3]

    TRAIN: [0 1 2 3] TEST: [4]

    TRAIN: [0 1 2 3 4] TEST: [5]

    """



    def __init__(self,

                 n_splits: Optional[int] = 5,

                 train_size: Optional[int] = None,

                 test_size: Optional[int] = None,

                 delay: int = 0,

                 force_step_size: Optional[int] = None):



        if n_splits and n_splits < 5:

            raise ValueError(f'Cannot have n_splits less than 5 (n_splits={n_splits})')

        super().__init__(n_splits, shuffle=False, random_state=None)



        self.train_size = train_size



        if test_size and test_size < 0:

            raise ValueError(f'Cannot have negative values of test_size (test_size={test_size})')

        self.test_size = test_size



        if delay < 0:

            raise ValueError(f'Cannot have negative values of delay (delay={delay})')

        self.delay = delay



        if force_step_size and force_step_size < 1:

            raise ValueError(f'Cannot have zero or negative values of force_step_size '

                             f'(force_step_size={force_step_size}).')



        self.force_step_size = force_step_size



    def split(self, X, y=None, groups=None):

        """Generate indices to split data into training and test set.



        Parameters:

            X : array-like, shape (n_samples, n_features)

                Training data, where n_samples is the number of samples  and n_features is the number of features.



            y : array-like, shape (n_samples,)

                Always ignored, exists for compatibility.



            groups : array-like, with shape (n_samples,), optional

                Always ignored, exists for compatibility.



        Yields:

            train : ndarray

                The training set indices for that split.



            test : ndarray

                The testing set indices for that split.

        """

        X, y, groups = indexable(X, y, groups)  # pylint: disable=unbalanced-tuple-unpacking

        n_samples = _num_samples(X)

        n_splits = self.n_splits

        n_folds = n_splits + 1

        delay = self.delay



        if n_folds > n_samples:

            raise ValueError(f'Cannot have number of folds={n_folds} greater than the number of samples: {n_samples}.')



        indices = np.arange(n_samples)

        split_size = n_samples // n_folds



        train_size = self.train_size or split_size * self.n_splits

        test_size = self.test_size or n_samples // n_folds

        full_test = test_size + delay



        if full_test + n_splits > n_samples:

            raise ValueError(f'test_size\\({test_size}\\) + delay\\({delay}\\) = {test_size + delay} + '

                             f'n_splits={n_splits} \n'

                             f' greater than the number of samples: {n_samples}. Cannot create fold logic.')



        # Generate logic for splits.

        # Overwrite fold test_starts ranges if force_step_size is specified.

        if self.force_step_size:

            step_size = self.force_step_size

            final_fold_start = n_samples - (train_size + full_test)

            range_start = (final_fold_start % step_size) + train_size



            test_starts = range(range_start, n_samples, step_size)



        else:

            if not self.train_size:

                step_size = split_size

                range_start = (split_size - full_test) + split_size + (n_samples % n_folds)

            else:

                step_size = (n_samples - (train_size + full_test)) // n_folds

                final_fold_start = n_samples - (train_size + full_test)

                range_start = (final_fold_start - (step_size * (n_splits - 1))) + train_size



            test_starts = range(range_start, n_samples, step_size)



        # Generate data splits.

        for test_start in test_starts:

            idx_start = test_start - train_size if self.train_size is not None else 0

            # Ensure we always return a test set of the same size

            if indices[test_start:test_start + full_test].size < full_test:

                continue

            yield (indices[idx_start:test_start],

                   indices[test_start + delay:test_start + full_test])
%%time

# From kernel https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

# WARNING! THIS CAN DAMAGE THE DATA 

def reduce_mem_usage2(df):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import gc

# Any results you write to the current directory are saved as output.

import imblearn

from imblearn.under_sampling import RandomUnderSampler,TomekLinks

import datetime

import lightgbm as lgb

import sklearn as skl

from sklearn.model_selection import *

from sklearn.metrics import *

import matplotlib.pyplot as plt

from imblearn.over_sampling import *

from sklearn.decomposition import PCA, TruncatedSVD, FastICA

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.utils import resample

from dateutil import relativedelta
%%time

folder_path = '../input/ieee-fraud-detection/'

train_id = pd.read_csv(f'{folder_path}train_identity.csv')

train_tr = pd.read_csv(f'{folder_path}train_transaction.csv')

test_id = pd.read_csv(f'{folder_path}test_identity.csv')

test_tr = pd.read_csv(f'{folder_path}test_transaction.csv')

sub = pd.read_csv(f'{folder_path}sample_submission.csv')

train = pd.merge(train_tr, train_id, on='TransactionID', how='left')

test = pd.merge(test_tr, test_id, on='TransactionID', how='left')

del train_id,test_id,train_tr,test_tr

gc.collect()

train = train.drop('id_31',axis=1)

test = test.drop('id_31',axis=1)
def make_time(df):

    START_DATE = '2017-11-30'

    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))



    #df['year'] = df['TransactionDT'].dt.year

    #df['month'] = df['TransactionDT'].dt.month

    df['dow'] = df['TransactionDT'].dt.dayofweek

    df['hour'] = df['TransactionDT'].dt.hour

    df['day'] = df['TransactionDT'].dt.day

    #df = df.drop(['TransactionDT','TransactionID'],axis=1)

    return df



def split_pseudo(df,gap=15):

    split_time_head = df['TransactionDT'][0]+relativedelta.relativedelta(days=+180/2-int(gap/2))

    split_time_tail = df['TransactionDT'][0]+relativedelta.relativedelta(days=+180/2+int(gap/2))

    train = df[df['TransactionDT']<split_time_head]

    test = df[df['TransactionDT']>split_time_tail]

    train = train.drop('TransactionDT',axis=1)

    test = test.drop('TransactionDT',axis=1)

    del df 

    gc.collect()

    return train,test

def clean_inf_nan(df):

    return df.replace([np.inf, -np.inf], np.nan)
train =make_time(train)

train = train.drop('TransactionID',axis=1)
test =make_time(test)

test = test.drop('TransactionID',axis=1)

test = test.drop('TransactionDT',axis=1)
train,pseudo_test =  split_pseudo(train)
%%time

train = clean_inf_nan(train)

pseudo_test = clean_inf_nan(pseudo_test)

test = clean_inf_nan(test)

train = reduce_mem_usage2(train)

pseudo_test = reduce_mem_usage2(pseudo_test)

test = reduce_mem_usage2(test)
y = train['isFraud']

train = train.drop(['isFraud'],axis=1)

y_pseudo = pseudo_test['isFraud']

pseudo_test = pseudo_test.drop(['isFraud'],axis=1)
pseudo_test_LB = pseudo_test.iloc[0:int(0.2*len(pseudo_test)),:]

y_LB = y_pseudo.iloc[0:int(0.2*len(pseudo_test))]
train.shape[1],pseudo_test.shape[1]
params  = {

          'num_leaves': 190,

          'num_boost_round':5000,

          'min_child_samples': 79,

          'objective': 'binary',

          'max_depth': 13,

          'learning_rate': 0.01,

          "boosting_type": "gbdt",

          "subsample_freq": 1,

          "subsample": 0.6,

          "metric": 'auc',

          #"verbosity": -1,

          'reg_alpha': 0.4,

          'reg_lambda': 0.4,

          'colsample_bytree': 0.3,

          #'categorical_feature': cat_cols

         }
test.columns
sub['isFraud'] = 0.
total_cv = 0.

total_p_cv = 0.

total_lb = 0.

n_folds=5

folds = TimeSeriesSplit_(delay=30,train_size=int(len(train)/2),test_size=int(len(train)/6)-100)



for fold_n,(train_idx,val_idx) in enumerate(folds.split(train)):

    tr_X,val_X = train.iloc[train_idx],train.iloc[val_idx]

    tr_y,val_y = y.iloc[train_idx],y.iloc[val_idx]

    

    lgtrain = lgb.Dataset(tr_X,label=tr_y)

    lgval = lgb.Dataset(val_X,label=val_y)

    

    print('fold: %d  len_train:%d len_val:%d '%(fold_n+1,len(train_idx),len(val_idx)))

    model = lgb.train(params,lgtrain,valid_sets=[lgtrain,lgval],

                      early_stopping_rounds=70,verbose_eval=200)

    ##########################

    print('#'*20)

    cv_predict = model.predict(val_X)

    cv_score = roc_auc_score(y_true=val_y,y_score=cv_predict)

    total_cv +=cv_score

    

    print('AUC on cv: %f'%(cv_score))

    ########################################

    p_cv_predict = model.predict(pseudo_test)

    p_cv_score = roc_auc_score(y_true=y_pseudo,y_score=p_cv_predict)

    total_p_cv+=p_cv_score

    print('AUC on p_cv: %f'%(p_cv_score))

    ##########################################

    

    LB_pre = model.predict(pseudo_test_LB)

    LB_score = roc_auc_score(y_true=y_LB,y_score=LB_pre)

    total_lb += LB_score

    print('pseudo LB auc score : %f'%(LB_score))

    print('#'*20)

    ########################################   

    pre = model.predict(test)

    print(pre.shape)

    sub['isFraud'] += pre

    ##################################

print('*'*20)

print('total_cv: %f'%(total_cv/n_folds))

print('total_p_cv: %f'%(total_p_cv/n_folds))

print('total_LB_auc: %f'%(total_lb/n_folds))
sub.head()
sub['isFraud'] /=n_folds

sub.to_csv('sub.csv',index=False)
test.columns