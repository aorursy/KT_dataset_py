# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

from time import time, ctime

from contextlib import contextmanager

import time

import gc



# boostings

import lightgbm as lgb

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from numba import jit



from sklearn import metrics

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import StratifiedKFold, KFold



import warnings

warnings.filterwarnings("ignore")
RANDOM_SEED = 42

train_path = '/kaggle/input/made-train/train.csv'

test_path = '/kaggle/input/madetest/test-data.tar'
def read_as_chunks(path, chunksize, sep=';', quotechar='"'):

    return pd.read_csv(path, header=0, sep=sep, quotechar=quotechar, chunksize=chunksize)
def memory_preprocess(chunk):

    chunk['timestamp'] = chunk['timestamp'].astype('int32')

    chunk['label'] = chunk['label'].astype('int8')

    chunk['C3'] = chunk['C3'].astype('int16')

    chunk['C4'] = chunk['C4'].astype('int16')

    chunk['C5'] = chunk['C5'].astype('int8')

    chunk['C6'] = chunk['C6'].astype('int16')

    chunk['C7'] = chunk['C7'].astype('int8')

    chunk['C8'] = chunk['C8'].astype('int16')

    chunk['C9'] = chunk['C9'].astype('int8')

    chunk['C10'] = chunk['C10'].astype('int16')

    chunk['l1'] = chunk['l1'].astype('int16')

    chunk['l2'] = chunk['l2'].astype('int8')

    chunk['C11'] = chunk['C11'].astype('int8')

    chunk['C12'] = chunk['C12'].astype('int8')

    return chunk
def preprocess(df):

    df.reset_index(drop=True, inplace=True)

    

    def create_category(base, i):

        return base + str(i)

#     categories_CG1 = [create_category('CG1_', i) for i in range(452)]

    df['CG1'].fillna('-1', inplace=True)

    df['CG1'] = df['CG1'].apply(lambda x: set(x.split(',')))

    for i in range(452):

        category = create_category('CG1_', i)

        num_to_search = str(i)

        df[category] = df['CG1'].apply(lambda x: num_to_search in x)

        df[category] = df[category].astype('int8')

    df.drop(['CG1', 'CG2', 'CG3', 'label'], axis = 1, inplace=True)

    return df
%%time

chunksize = 1000000

df_chunks = read_as_chunks(train_path, chunksize)

chunk_list = []

count_passed = 0

lgb_estimator = None

lgb_params = {

    'keep_training_booster': True,

    'objective': 'cross_entropy',

    'verbosity': 100,

    'n_estimators': 1200,

    'seed': RANDOM_SEED,

    'max_depth': 8

}

for chunk in df_chunks:

    chunk = memory_preprocess(chunk)

    y_train = chunk['label']

    processed_chunk = preprocess(chunk)

    lgb_estimator = lgb.train(lgb_params,

                             init_model = lgb_estimator,

                             train_set = lgb.Dataset(processed_chunk, y_train),

                             num_boost_round = 10)

    count_passed += 1

    del chunk, processed_chunk, y_train

    gc.collect()

    print("Passed step: " + str(count_passed))
test = pd.read_csv(test_path, header=0, sep=',', quotechar='"', index_col=0)

test.rename(columns = {'test.csv': 'timestamp'}, inplace=True)

test = memory_preprocess(test)

test = preprocess(test)

predictions = lgb_estimator.predict(test)

submission_df = pd.Series(predictions)

submission_df.to_csv('lgb_third.csv', index=False)