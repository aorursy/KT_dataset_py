# !pip install tensorflow_addons

# import tensorflow as tf

# from tensorflow.keras.layers import *

import pandas as pd

import numpy as np

import random

# from tensorflow.keras.callbacks import Callback, LearningRateScheduler

# from tensorflow.keras.losses import categorical_crossentropy

# from tensorflow.keras.optimizers import Adam

# from tensorflow.keras import backend as K

# from tensorflow.keras import losses, models, optimizers

# import tensorflow_addons as tfa

import gc



from sklearn.model_selection import GroupKFold

from sklearn.metrics import f1_score



import warnings

warnings.simplefilter('ignore')

warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 1000)

pd.set_option('display.max_rows', 500)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# configurations and main hyperparammeters

EPOCHS = 180

NNBATCHSIZE = 16

GROUP_BATCH_SIZE = 4000

SEED = 321

LR = 0.0015

SPLITS = 6



def seed_everything(seed):

    random.seed(seed)

    np.random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    tf.random.set_seed(seed)
# read data

def read_data():

    train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})

    test  = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})

    sub  = pd.read_csv('/kaggle/input/liverpool-ion-switching/sample_submission.csv', dtype={'time': np.float32})

    

    Y_train_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy")

    Y_test_proba = np.load("/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy")

    

    for i in range(11):

        train[f"proba_{i}"] = Y_train_proba[:, i]

        test[f"proba_{i}"] = Y_test_proba[:, i]



    return train, test, sub



# create batches of 4000 observations

def batching(df, batch_size):

    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values

    df['group'] = df['group'].astype(np.uint16)

    return df



# normalize the data (standard scaler). We can also try other scalers for a better score!

def normalize(train, test):

    train_input_mean = train.signal.mean()

    train_input_sigma = train.signal.std()

    train['signal'] = (train.signal - train_input_mean) / train_input_sigma

    test['signal'] = (test.signal - train_input_mean) / train_input_sigma

    return train, test



# get lead and lags features

def lag_with_pct_change(df, windows):

    for window in windows:    

        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)

        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)

    return df



# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).

def run_feat_engineering(df, batch_size):

    # create batches

    df = batching(df, batch_size = batch_size)

    # create leads and lags (1, 2, 3 making them 6 features)

    df = lag_with_pct_change(df, [1, 2, 3])

    # create signal ** 2 (this is the new feature)

    df['signal_2'] = df['signal'] ** 2

    return df



# fillna with the mean and select features for training

def feature_selection(train, test):

    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]

    train = train.replace([np.inf, -np.inf], np.nan)

    test = test.replace([np.inf, -np.inf], np.nan)

    for feature in features:

        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()

        train[feature] = train[feature].fillna(feature_mean)

        test[feature] = test[feature].fillna(feature_mean)

    return train, test, features
train, test, sample_submission = read_data()

train, test = normalize(train, test)
train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)

test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)

train, test, features = feature_selection(train, test)
train['signal_rolling_mean_1h'] = train['signal'].rolling(window = 100).mean().fillna(0)

test['signal_rolling_mean_1h'] = test['signal'].rolling(window = 100).mean().fillna(0)



train['signal_rolling_std_1h'] = train['signal'].rolling(window = 100).std().fillna(0)

test['signal_rolling_std_1h'] = test['signal'].rolling(window = 100).std().fillna(0)



train['signal_rolling_mean_1t'] = train['signal'].rolling(window = 1000).mean().fillna(0)

test['signal_rolling_mean_1t'] = test['signal'].rolling(window = 1000).mean().fillna(0)



train['signal_rolling_std_1t'] = train['signal'].rolling(window = 1000).std().fillna(0)

test['signal_rolling_std_1t'] = test['signal'].rolling(window = 1000).std().fillna(0)
# train.head()

# test.head()

# features
from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score

# import lightgbm as lgb

import optuna.integration.lightgbm as lgb

import optuna, os, uuid, pickle
X = train.drop(['time', 'open_channels'], axis = 1)

y = train['open_channels']

X_test = test.drop(['time'], axis = 1)

X_train, X_val, y_train, y_val = train_test_split(X, y)



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)



params = {

    "objective": "multiclass",

    "num_class": 11,

    "metric": "multi_logloss",

    "verbosity": -1,

    "boosting_type": "gbdt",

#     "boosting_type": "dart",

#     "boosting_type": "goss",

    'num_boost_round': 3,

}



best_params, tuning_history = dict(), list()

model = lgb.train(

    params,

    lgb_train,

    valid_sets=[lgb_train, lgb_eval],

#     verbose_eval=1,

    verbose_eval=100,

    early_stopping_rounds=100,

#     early_stopping_rounds=1,

    best_params=best_params,

    tuning_history=tuning_history,

)



print("Best Params:", best_params)

print("Tuning history:", tuning_history)
best_params = model.params

best_params
model = lgb.train(best_params, lgb_train, valid_sets=lgb_eval, num_boost_round=3)

y_pred = model.predict(X_test, num_iteration=model.best_iteration)

y_pred_max = np.argmax(y_pred, axis=1)
sample_submission['open_channels'] = y_pred_max

# sample_submission['open_channels'].value_counts()



sample_submission.to_csv('submission_wavenet_lgbm.csv', index=False, float_format='%.4f')