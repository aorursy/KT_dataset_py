import os

import time

import math

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import GroupKFold

import lightgbm as lgb

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

sns.set()



print("Files in the input folder:")

print(os.listdir("../input"))

train = pd.read_csv('../input/X_train.csv')

test = pd.read_csv('../input/X_test.csv')

y = pd.read_csv('../input/y_train.csv')

sub = pd.read_csv('../input/sample_submission.csv')

print("\nX_train shape: {}, X_test shape: {}".format(train.shape, test.shape))

print("y_train shape: {}, submission shape: {}".format(y.shape, sub.shape))
train.head()
y.head()
plt.figure(figsize=(10,6))

plt.title("Training labels")

ax = sns.countplot(y='surface', data=y)
y.groupby('group_id').surface.nunique().max()
plt.figure(figsize=(22,6)) 

sns.countplot(x='group_id', data=y, order=y.group_id.value_counts().index)

plt.show()
data = train.merge(y, on='series_id', how='left')



# Some utility functions here



def plot_box_and_kde(num_axis, feature_group, catplot='boxplot'):

    plt.figure()

    fig, ax = plt.subplots(num_axis, 2,figsize=(12, 5 * num_axis))

    j = 0

    for i in range(num_axis):

        axis_list = ['_X', '_Y', '_Z', '_W']

        col = feature_group + axis_list[i]

        j += 1

        plt.subplot(num_axis, 2, j)

        if catplot == 'boxplot':

            sns.boxplot(y='surface', x=col, data=data, fliersize=0.4)

        else:

            sns.violinplot(y='surface', x=col, data=data)

        j += 1

        plt.subplot(num_axis, 2, j)

        sns.kdeplot(train[col], label='train')

        sns.kdeplot(test[col], label='test')

    plt.show()





def quaternion_to_euler(x, y, z, w):

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)

    return X, Y, Z

    

plot_box_and_kde(3, 'linear_acceleration')
plot_box_and_kde(3, 'angular_velocity')
def convert_to_euler(df):

    euler = df.apply(lambda r: quaternion_to_euler(r['orientation_X'], r['orientation_Y'],

                                                   r['orientation_Z'], r['orientation_W']), axis=1)

    df['euler_X'] = np.array([value[0] for value in euler])

    df['euler_Y'] = np.array([value[1] for value in euler])

    df['euler_Z'] = np.array([value[2] for value in euler])

    return df



data = convert_to_euler(data)

train = convert_to_euler(train)

test = convert_to_euler(test)
plot_box_and_kde(3, 'euler')
def change1(x):

    return np.mean(np.abs(np.diff(x)))



def change2(x):

    return np.mean(np.diff(np.abs(np.diff(x))))



def feature_extraction(df):

    feat = pd.DataFrame()

    df['linear_acceleration'] = np.sqrt(df['linear_acceleration_X']**2 + df['linear_acceleration_Y']**2 + df['linear_acceleration_Z']**2)

    df['linear_acceleration_XZ'] = np.sqrt(df['linear_acceleration_X']**2 + df['linear_acceleration_Z']**2)

    

    df['acceleration_X_cumsum'] = df['linear_acceleration_X'].cumsum().fillna(0)

    df['acceleration_Y_cumsum'] = df['linear_acceleration_Y'].cumsum().fillna(0)

    df['acceleration_Z_cumsum'] = df['linear_acceleration_Z'].cumsum().fillna(0)

    

    for col in df.columns[3:]:

        feat[col + '_mean'] = df.groupby(['series_id'])[col].mean()

        feat[col + '_std'] = df.groupby(['series_id'])[col].std()

        feat[col + '_max'] = df.groupby(['series_id'])[col].max()

        feat[col + '_min'] = df.groupby(['series_id'])[col].min()

        feat[col + '_max_to_min'] = feat[col + '_max'] / feat[col + '_min']

        

        # Change 1st order

        feat[col + '_mean_abs_change'] = df.groupby('series_id')[col].apply(change1)

        # Change 2nd order

        feat[col + '_mean_abs_change2'] = df.groupby('series_id')[col].apply(change2)

        feat[col + '_abs_max'] = df.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

    return feat
train_df = feature_extraction(train)

test_df = feature_extraction(test)

train_df.head()
train_df.shape
le = LabelEncoder()

target = le.fit_transform(y['surface'])
params = {

    'num_leaves': 18,

    'min_data_in_leaf': 40,

    'objective': 'multiclass',

    'metric': 'multi_error',

    'max_depth': 8,

    'learning_rate': 0.01,

    "boosting": "gbdt",

    "bagging_freq": 5,

    "bagging_fraction": 0.812667,

    "bagging_seed": 11,

    "verbosity": -1,

    'reg_alpha': 0.3,

    'reg_lambda': 0.1,

    "num_class": 9,

    'nthread': -1

}



t0 = time.time()

train_set = lgb.Dataset(train_df, label=target)

eval_hist = lgb.cv(params, train_set, nfold=10, num_boost_round=9999,

                   early_stopping_rounds=100, seed=19)

num_rounds = len(eval_hist['multi_error-mean'])

# retrain the model and make predictions for test set

clf = lgb.train(params, train_set, num_boost_round=num_rounds)

predictions = clf.predict(test_df, num_iteration=None)

print("Timer: {:.1f}s".format(time.time() - t0))
v = eval_hist['multi_error-mean'][-1]

print("Validation error: {:.4f}, accuracy: {:.4f}".format(v, 1 - v))

plt.figure(figsize=(10, 4))

plt.title("CV multiclass error")

num_rounds = len(eval_hist['multi_error-mean'])

ax = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_error-mean'])

ax2 = ax.twinx()

p = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_error-stdv'], ax=ax2, color='r')
importance = pd.DataFrame({'gain': clf.feature_importance(importance_type='gain'),

                           'feature': clf.feature_name()})

importance.sort_values(by='gain', ascending=False, inplace=True)

plt.figure(figsize=(10, 28))

ax = sns.barplot(x='gain', y='feature', data=importance)
sub['surface'] = le.inverse_transform(predictions.argmax(axis=1))

sub.to_csv('submission_kfold.csv', index=False)

sub.surface.value_counts()
group_info = pd.DataFrame()

group_info['num_groups'] = y.groupby('surface').group_id.nunique()

group_info['num_samples'] = y.groupby('surface').size()

group_info
num_folds = 5



def group_kfold():

    """Generator that yiels train and test indexes."""

    folds = GroupKFold(n_splits=num_folds)

    for train_idx, test_idx in folds.split(train_df, groups=y['group_id'].values):

        yield train_idx, test_idx





t0 = time.time()

train_set = lgb.Dataset(train_df, label=target)

eval_hist = lgb.cv(params, train_set, nfold=num_folds, num_boost_round=9999, folds=group_kfold(),

                   early_stopping_rounds=100, seed=19)

num_rounds = len(eval_hist['multi_error-mean'])

clf = lgb.train(params, train_set, num_boost_round=num_rounds)

predictions = clf.predict(test_df, num_iteration=None)

print("Timer: {:.1f}s".format(time.time() - t0))

v = eval_hist['multi_error-mean'][-1]

print("Validation error: {:.4f}, accuracy: {:.4f}".format(v, 1 - v))
plt.figure(figsize=(10, 4))

plt.title("CV multiclass error")

num_rounds = len(eval_hist['multi_error-mean'])

ax = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_error-mean'])

ax2 = ax.twinx()

p = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_error-stdv'], ax=ax2, color='r')
sub['surface'] = le.inverse_transform(predictions.argmax(axis=1))

sub.to_csv('submission_group.csv', index=False)

sub.surface.value_counts()
num_folds = 72



def logo_cv():

    """Generator that yiels train and test indexes."""

    for group_id in range(73):

        if group_id == 27: continue

        test_idx = list(y[y.group_id == group_id].index)

        train_idx = [i for i in range(3810) if i not in test_idx]

        yield train_idx, test_idx



t0 = time.time()

train_set = lgb.Dataset(train_df, label=target)

eval_hist = lgb.cv(params, train_set, nfold=num_folds, num_boost_round=9999, folds=logo_cv(),

                   early_stopping_rounds=100, seed=19)

num_rounds = len(eval_hist['multi_error-mean'])

clf = lgb.train(params, train_set, num_boost_round=num_rounds)

predictions = clf.predict(test_df, num_iteration=None)

print("Timer: {:.1f}s".format(time.time() - t0))

v = eval_hist['multi_error-mean'][-1]

print("Validation error: {:.4f}, accuracy: {:.4f}".format(v, 1 - v))
plt.figure(figsize=(10, 4))

plt.title("CV multiclass error")

num_rounds = len(eval_hist['multi_error-mean'])

ax = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_error-mean'])

ax2 = ax.twinx()

p = sns.lineplot(x=range(num_rounds), y=eval_hist['multi_error-stdv'], ax=ax2, color='r')
sub['surface'] = le.inverse_transform(predictions.argmax(axis=1))

sub.to_csv('submission_logo.csv', index=False)

sub.surface.value_counts()