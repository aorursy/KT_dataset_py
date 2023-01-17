import time

import random

import numpy as np

import pandas as pd

from tqdm import tqdm_notebook

# seaborn and matplot

import matplotlib.pyplot as plt

import seaborn as sns

# scipy (feature engineering)

from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats

import lightgbm as lgb

import warnings

# Configurations

warnings.simplefilter(action='ignore', category=UserWarning)

RANDOM_SEED = 19

np.random.seed(RANDOM_SEED)

sns.set()



def plot_multiclass(result):

    # Plot multi_logloss and 1 - multi_error

    num_rounds = len(result['multi_logloss-mean'])

    fig, ax1 = plt.subplots(figsize=(10, 5))

    fig.suptitle('logloss (blue) and accuracy (orange)', fontsize=14)

    ax2 = ax1.twinx()

    ax1.set_xlabel('boosting round')

    ax1.set_ylabel('logloss')

    ax2.set_ylabel('accuracy')

    p1 = sns.lineplot(x=np.arange(num_rounds), y=result['multi_logloss-mean'],

                      ax=ax1, color='blue')

    multi_accuracy = [1 - v for v in result['multi_error-mean']]  # not sure if this is right

    p2 = sns.lineplot(x=np.arange(num_rounds), y=multi_accuracy, ax=ax2, color='orange')
data_type = {'acoustic_data': np.int16, 'time_to_failure': np.float64}

train = pd.read_csv('../input/train.csv', dtype=data_type)
def extract_segment_features(frame, index, x):

    frame.loc[index, 'std'] = x.values.std()

    frame.loc[index, 'mean'] = x.values.mean()

    frame.loc[index, 'max'] = x.values.max()

    frame.loc[index, 'min'] = x.values.min()

    frame.loc[index, 'std_abs'] = x.abs().std()

    frame.loc[index, 'max_abs'] = x.abs().max()

    frame.loc[index, 'mean_abs_change'] = np.mean(np.abs(x.diff()))

    frame.loc[index, 'std_abs_change'] = np.std(np.abs(x.diff()))

    

    frame.loc[index, 'mad'] = x.mad()

    frame.loc[index, 'iqr'] = stats.iqr(x.values)

    frame.loc[index, 'kurt'] = x.kurtosis()

    frame.loc[index, 'skew'] = x.skew()

    frame.loc[index, 'q05'] = np.quantile(x, 0.05)

    frame.loc[index, 'q95'] = np.quantile(x, 0.95)

    

    for windows in [16, 64, 512, 4096]:

        x_roll_mean = x.rolling(windows).mean().dropna().values

        x_roll_std = x.rolling(windows).std().dropna().values

        frame.loc[index, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)   

        frame.loc[index, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        frame.loc[index, 'mean_roll_std_' + str(windows)] = np.mean(x_roll_mean)

        frame.loc[index, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

    return frame





def make_train(train_data, size=150000, skip=150000):

    num_segments = int(np.floor((train_data.shape[0] - size) / skip)) + 1

    # We will be removing segments that belongs to two quakes

    num_segments -= 16



    X_train = pd.DataFrame(index=range(num_segments), dtype=np.float64)

    y_train = pd.DataFrame(index=range(num_segments), columns=['quake_number'])

    quake_count = 0

    

    for index in tqdm_notebook(range(num_segments)):

        seg = train_data.iloc[index*skip:index*skip + size]

        

        if any(seg.time_to_failure.diff() > 5):

            quake_count += 1

            continue

        

        y_train.loc[index, 'quake_number'] = quake_count

        y_train.loc[index, 'time_to_failure'] = seg.time_to_failure.values[-1]

        X_train = extract_segment_features(X_train, index, seg.acoustic_data)

    return X_train, y_train
X_tr, y_tr = make_train(train)

X_tr.head()
plt.figure(figsize=(10, 5))

plt.title("Correlation heatmap for features")

ax = sns.heatmap(X_tr.corr(), annot=False, linewidths=.3, cmap="YlGnBu")

plt.figure(figsize=(10, 5))

plt.title("Count of segments for each earthquake (y_tr)")

ax = sns.countplot(x="quake_number", data=y_tr, palette='GnBu_d')
keep_idx = y_tr[(y_tr.quake_number > 0) & (y_tr.quake_number < 16)].index

X_tr, y_tr = X_tr.iloc[keep_idx], y_tr.iloc[keep_idx]

y_tr.quake_number = y_tr.quake_number - 1  # start counting at 0

plt.figure(figsize=(10, 5))

plt.title("New count (y_tr)")

ax = sns.countplot(x="quake_number", data=y_tr, palette='GnBu_d')
params = {

    'objective': 'multiclass',  # Softmax

    'metric': ['multi_logloss', 'multi_error'],

    'num_class': 15,

    "boosting": "gbdt",

    'num_leaves': 32,

    'min_data_in_leaf': 10, 

    'max_depth': -1,

    'learning_rate': 0.01,

    "feature_fraction": 1,

    "bagging_freq": 5,

    "bagging_fraction": 0.9,

    "bagging_seed": 19,

    "lambda_l2": 0.1,

    "num_boost_round": 90000,

    "verbosity": -1,

    "nthread": -1,

}

dataset = lgb.Dataset(X_tr, label=y_tr.quake_number)

result = lgb.cv(params, dataset, nfold=10, early_stopping_rounds=100, stratified=False)

plot_multiclass(result)
params['num_boost_round'] = len(result['multi_error-mean'])

bst = lgb.train(params, dataset)

s = pd.DataFrame({'feature': X_tr.columns,

                  'gain': bst.feature_importance(importance_type='gain')})

s.sort_values(by='gain', ascending=False).head()
params['num_boost_round'] = 99999

params['num_class'] = 3

idx = y_tr[(y_tr.quake_number >= 0) & (y_tr.quake_number < 3)].index

dataset = lgb.Dataset(X_tr.iloc[idx], label=y_tr.loc[idx, 'quake_number'])

result = lgb.cv(params, dataset, nfold=10, early_stopping_rounds=100, stratified=False)

plot_multiclass(result)
params = {

    'objective': 'binary',

    'metric': 'auc',

    "boosting": "gbdt",

    'num_leaves': 22,

    'min_data_in_leaf': 10, 

    'max_depth': -1,

    'learning_rate': 0.01,

    "feature_fraction": 1,

    "bagging_freq": 5,

    "bagging_fraction": 0.9,

    "bagging_seed": 19,

    "lambda_l2": 0.05,

    "num_boost_round": 90000,

    "verbosity": -1,

    "nthread": -1,

}

# Predict if a segment came from group A or B

def binary_prediction(q1, q2):

    assert(q1 < q2)

    idx = y_tr[(y_tr.quake_number == q1) | (y_tr.quake_number == q2)].index

    binary_target = y_tr.loc[idx, 'quake_number'] > q1

    dataset = lgb.Dataset(X_tr.iloc[idx], label=binary_target)

    result = lgb.cv(params, dataset, nfold=10, early_stopping_rounds=100, stratified=True)

    num_rounds = len(result['auc-mean'])



    plt.figure(figsize=(10, 5))

    plt.title("AUC - earthquake {} vs {}".format(q1, q2))

    ax = sns.lineplot(x=np.arange(num_rounds), y=result['auc-mean'])
binary_prediction(2, 3)
binary_prediction(2, 11)
binary_prediction(1, 6)
binary_target = (y_tr.time_to_failure > 12).astype('int8')

binary_target.value_counts()
f = X_tr.columns

dataset = lgb.Dataset(X_tr[f], label=binary_target)

result = lgb.cv(params, dataset, nfold=10, early_stopping_rounds=100,

                stratified=False, shuffle=True)

num_rounds = len(result['auc-mean'])



plt.figure(figsize=(10, 5))

plt.title("AUC - predicting TTF > 12")

ax = sns.lineplot(x=np.arange(num_rounds), y=result['auc-mean'])
f = ['q05_roll_std_64', 'q95_roll_mean_64']

dataset = lgb.Dataset(X_tr[f], label=binary_target)

result = lgb.cv(params, dataset, nfold=10, early_stopping_rounds=100,

                stratified=False, shuffle=True)

num_rounds = len(result['auc-mean'])



plt.figure(figsize=(10, 5))

plt.title("AUC - predicting TTF > 12")

ax = sns.lineplot(x=np.arange(num_rounds), y=result['auc-mean'])