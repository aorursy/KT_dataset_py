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
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import log_loss

from datetime import datetime

from lightgbm import LGBMClassifier

from xgboost import XGBClassifier

import gc
import keras

import tensorflow as tf

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense, Dropout, Activation, BatchNormalization, LeakyReLU

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from tensorflow.keras.models import load_model
# random seed fixed

np.random.seed(5)

tf.compat.v1.set_random_seed(5)
# Read training data

train = pd.read_csv('/kaggle/input/dacon-stage-2/train.csv', index_col='id')

train.tail(2)
# Read test data

test = pd.read_csv('/kaggle/input/dacon-stage-2/test.csv', index_col='id')

test.tail(2)
train.shape, test.shape
train.describe().T
test.describe().T
train.groupby('type').mean().T
train['fiberID'].value_counts()
test['fiberID'].value_counts()
plt.figure(figsize=(30, 7))

sns.countplot(x=train['type'])
plt.figure(figsize=(30, 7))

sns.boxplot(train['type'], train['fiberID'])
plt.figure(figsize=(25, 7))

sns.boxplot(train['type'], train['petroMag_z'])
plt.figure(figsize=(25, 7))

sns.boxplot(train['type'], train['psfMag_g'])
plt.figure(figsize=(10, 8))

sns.heatmap(train.corr())
def draw_types(n=6, regex='psf'):

    labels = train['type'].value_counts().index.tolist()[:n]

    columns = train.filter(regex=regex).columns.tolist()

    colors = ['violet', 'green', 'red', 'cyan', 'yellow']

    waves = [column[-1:] for column in columns]



    fig, axes = plt.subplots(int(n/2), 2, figsize=(10,n), dpi=100)

    w = 1.5

    for i, label in enumerate(labels):

        for column, color, wave in zip(columns, colors, waves):

            q1 = train.loc[train['type'] == label, column].quantile(0.25)

            q3 = train.loc[train['type'] == label, column].quantile(0.75)

            iqr = q3 - q1

            lower_bound = q1 - (w * iqr)

            upper_bound = q3 + (w * iqr)

            mask = (train.loc[train['type'] == label, column] >= lower_bound)

            mask = (train.loc[train['type'] == label, column] <= upper_bound)

            data = train.loc[train['type'] == label, column].loc[mask]



            sns.distplot(data, hist=False, color=color, kde_kws={'shade': True}, 

                         label=wave, ax=axes.flat[i])



        axes.flat[i].set_title(label)

        axes.flat[i].set_xlabel('')

        axes.flat[i].grid(axis='x', linestyle='--')

        axes.flat[i].legend(frameon=True, framealpha=1, shadow=False, 

                            fancybox=False, edgecolor='black')



    fig.tight_layout()

    plt.show()
draw_types(n=4, regex='psf')
draw_types(n=4, regex='fiberMag')
draw_types(n=4, regex='petroMag')
draw_types(n=4, regex='modelMag')
# Adjust training data range

train = train[train['psfMag_u'] < 40]

train = train[train['psfMag_g'] < 185]

train = train[train['psfMag_r'] < 35]

train = train[train['psfMag_i'] < 50]

train = train[train['psfMag_z'] < 35]

train = train[train['fiberMag_u'] < 45]

train = train[train['fiberMag_g'] < 50]

train = train[train['fiberMag_r'] < 30]

train = train[train['fiberMag_i'] < 35]

train = train[train['fiberMag_z'] < 30]

train = train[train['petroMag_u'] < 70]

train = train[train['petroMag_g'] < 110]

train = train[train['petroMag_r'] < 45]

train = train[train['petroMag_i'] < 55]

train = train[train['petroMag_z'] < 75]

train = train[train['modelMag_u'] < 35]

train = train[train['modelMag_g'] < 30]

train = train[train['modelMag_r'] < 30]

train = train[train['modelMag_i'] < 30]

train = train[train['modelMag_z'] < 25]



train = train[train['psfMag_u'] > -10]

train = train[train['psfMag_g'] > -45]

train = train[train['psfMag_r'] > 5]

train = train[train['psfMag_i'] > -25]

train = train[train['psfMag_z'] > 10]

train = train[train['fiberMag_u'] > 5]

train = train[train['fiberMag_g'] > 5]

train = train[train['fiberMag_r'] > 10]

train = train[train['fiberMag_i'] > 10]

train = train[train['fiberMag_z'] > -10]

train = train[train['petroMag_u'] > -100]

train = train[train['petroMag_g'] > -1350]

train = train[train['petroMag_r'] > -25]

train = train[train['petroMag_i'] > -10]

train = train[train['petroMag_z'] > -70]

train = train[train['modelMag_u'] > 10]

train = train[train['modelMag_g'] > 10]

train = train[train['modelMag_r'] > 10]

train = train[train['modelMag_i'] > 10]

train = train[train['modelMag_z'] > 10]



train.shape
def col_mean(u, g, r, i, z):

    return np.mean([u, g, r, i, z])

def col_median(u, g, r, i, z):

    return np.median([u, g, r, i, z])

def col_std(u, g, r, i, z):

    return np.std([u, g, r, i, z])

def col_sum(u, g, r, i, z):

    return np.sum([u, g, r, i, z])   



# train['psfMag_mean'] = train.apply(lambda x : col_mean(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)

# train['psfMag_median'] = train.apply(lambda x : col_median(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)

# train['psfMag_std'] = train.apply(lambda x : col_std(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)

# train['psfMag_sum'] = train.apply(lambda x : col_sum(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)



# train['fiberMag_mean'] = train.apply(lambda x : col_mean(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)

# train['fiberMag_median'] = train.apply(lambda x : col_median(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)

# train['fiberMag_std'] = train.apply(lambda x : col_std(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)

# train['fiberMag_sum'] = train.apply(lambda x : col_sum(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)



# train['petroMag_mean'] = train.apply(lambda x : col_mean(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)

# train['petroMag_median'] = train.apply(lambda x : col_median(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)

# train['petroMag_std'] = train.apply(lambda x : col_std(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)

# train['petroMag_sum'] = train.apply(lambda x : col_sum(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)



train['modelMag_mean'] = train.apply(lambda x : col_mean(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)

train['modelMag_median'] = train.apply(lambda x : col_median(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)

# train['modelMag_std'] = train.apply(lambda x : col_std(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)

train['modelMag_sum'] = train.apply(lambda x : col_sum(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)



train.head(2)
# test['psfMag_mean'] = test.apply(lambda x : col_mean(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)

# test['psfMag_median'] = test.apply(lambda x : col_median(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)

# test['psfMag_std'] = test.apply(lambda x : col_std(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)

# test['psfMag_sum'] = test.apply(lambda x : col_sum(x['psfMag_u'], x['psfMag_g'], x['psfMag_r'], x['psfMag_i'], x['psfMag_z']), axis=1)



# test['fiberMag_mean'] = test.apply(lambda x : col_mean(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)

# test['fiberMag_median'] = test.apply(lambda x : col_median(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)

# test['fiberMag_std'] = test.apply(lambda x : col_std(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)

# test['fiberMag_sum'] = test.apply(lambda x : col_sum(x['fiberMag_u'], x['fiberMag_g'], x['fiberMag_r'], x['fiberMag_i'], x['fiberMag_z']), axis=1)



# test['petroMag_mean'] = test.apply(lambda x : col_mean(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)

# test['petroMag_median'] = test.apply(lambda x : col_median(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)

# test['petroMag_std'] = test.apply(lambda x : col_std(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)

# test['petroMag_sum'] = test.apply(lambda x : col_sum(x['petroMag_u'], x['petroMag_g'], x['petroMag_r'], x['petroMag_i'], x['petroMag_z']), axis=1)



test['modelMag_mean'] = test.apply(lambda x : col_mean(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)

test['modelMag_median'] = test.apply(lambda x : col_median(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)

# test['modelMag_std'] = test.apply(lambda x : col_std(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)

test['modelMag_sum'] = test.apply(lambda x : col_sum(x['modelMag_u'], x['modelMag_g'], x['modelMag_r'], x['modelMag_i'], x['modelMag_z']), axis=1)



test.head(2)
def ugri_mean(u, g, r, i):

    return np.mean([u, g, r, i])

def ugri_max(u, g, r, i):

    return np.max([u, g, r, i])

def ugri_min(u, g, r, i):

    return np.min([u, g, r, i])



train['i_mean'] = train.apply(lambda x : ugri_mean(x['psfMag_i'], x['fiberMag_i'], 

                                                                 x['petroMag_i'], x['modelMag_i']), axis=1)



train['g_max'] = train.apply(lambda x : ugri_max(x['psfMag_g'], x['fiberMag_g'], 

                                                                 x['petroMag_g'], x['modelMag_g']), axis=1)



train['r_max'] = train.apply(lambda x : ugri_max(x['psfMag_r'], x['fiberMag_r'], 

                                                                 x['petroMag_r'], x['modelMag_r']), axis=1)



train['g_min'] = train.apply(lambda x : ugri_min(x['psfMag_g'], x['fiberMag_g'], 

                                                                 x['petroMag_g'], x['modelMag_g']), axis=1)



train.head(1)
test['i_mean'] = test.apply(lambda x : ugri_mean(x['psfMag_i'], x['fiberMag_i'], 

                                                                 x['petroMag_i'], x['modelMag_i']), axis=1)



test['g_max'] = test.apply(lambda x : ugri_max(x['psfMag_g'], x['fiberMag_g'], 

                                                                 x['petroMag_g'], x['modelMag_g']), axis=1)



test['r_max'] = test.apply(lambda x : ugri_max(x['psfMag_r'], x['fiberMag_r'], 

                                                                 x['petroMag_r'], x['modelMag_r']), axis=1)



test['g_min'] = test.apply(lambda x : ugri_min(x['psfMag_g'], x['fiberMag_g'], 

                                                                 x['petroMag_g'], x['modelMag_g']), axis=1)



test.head(1)
# did not use these features.

# # Statistics and distance based features 

# gb = train.groupby('fiberID', as_index=False).agg({'fiberMag_u': {'fiberMag_u_max_fiberID': np.max, 'fiberMag_u_min_fiberID': np.min, 

#                                                                   'fiberMag_u_mean_fiberID' : np.mean, 'fiberMag_u_median_fiberID' : np.median,

#                                                                   'fiberMag_u_std_fiberID' : np.std},

#                                                    'fiberMag_g': {'fiberMag_g_max_fiberID': np.max, 'fiberMag_g_min_fiberID': np.min, 

#                                                                   'fiberMag_g_mean_fiberID' : np.mean, 'fiberMag_g_median_fiberID' : np.median,

#                                                                   'fiberMag_g_std_fiberID' : np.std},

#                                                    'fiberMag_r': {'fiberMag_r_max_fiberID': np.max, 'fiberMag_r_min_fiberID': np.min, 

#                                                                   'fiberMag_r_mean_fiberID' : np.mean, 'fiberMag_r_median_fiberID' : np.median,

#                                                                   'fiberMag_r_std_fiberID' : np.std},

#                                                    'fiberMag_i': {'fiberMag_i_max_fiberID': np.max, 'fiberMag_i_min_fiberID': np.min, 

#                                                                   'fiberMag_i_mean_fiberID' : np.mean, 'fiberMag_i_median_fiberID' : np.median,

#                                                                   'fiberMag_i_std_fiberID' : np.std},

#                                                    'fiberMag_z': {'fiberMag_z_max_fiberID': np.max, 'fiberMag_z_min_fiberID': np.min, 

#                                                                   'fiberMag_z_mean_fiberID' : np.mean, 'fiberMag_z_median_fiberID' : np.median,

#                                                                   'fiberMag_z_std_fiberID' : np.std},

#                                                    'fiberMag_mean' : {'fiberMag_mean_max_fiberID': np.max, 'fiberMag_mean_min_fiberID': np.min, 

#                                                                   'fiberMag_mean_mean_fiberID' : np.mean, 'fiberMag_mean_median_fiberID' : np.median,

#                                                                   'fiberMag_mean_std_fiberID' : np.std},

#                                                    'fiberMag_median' : {'fiberMag_median_max_fiberID': np.max, 'fiberMag_median_min_fiberID': np.min, 

#                                                                   'fiberMag_median_mean_fiberID' : np.mean, 'fiberMag_median_median_fiberID' : np.median,

#                                                                   'fiberMag_median_std_fiberID' : np.std},

#                                                    'fiberMag_std' : {'fiberMag_std_max_fiberID': np.max, 'fiberMag_std_min_fiberID': np.min, 

#                                                                   'fiberMag_std_mean_fiberID' : np.mean, 'fiberMag_std_median_fiberID' : np.median,

#                                                                   'fiberMag_std_std_fiberID' : np.std},

#                                                    'fiberMag_sum' : {'fiberMag_sum_max_fiberID': np.max, 'fiberMag_sum_min_fiberID': np.min, 

#                                                                   'fiberMag_sum_mean_fiberID' : np.mean, 'fiberMag_sum_median_fiberID' : np.median,

#                                                                   'fiberMag_sum_std_fiberID' : np.std}

#                                                    })



# gb.columns = ['fiberID', 'fiberMag_u_max_fiberID', 'fiberMag_u_min_fiberID', 'fiberMag_u_mean_fiberID', 'fiberMag_u_median_fiberID', 'fiberMag_u_std_fiberID',

#               'fiberMag_g_max_fiberID', 'fiberMag_g_min_fiberID', 'fiberMag_g_mean_fiberID', 'fiberMag_g_median_fiberID', 'fiberMag_g_std_fiberID',

#               'fiberMag_r_max_fiberID', 'fiberMag_r_min_fiberID', 'fiberMag_r_mean_fiberID', 'fiberMag_r_median_fiberID', 'fiberMag_r_std_fiberID',

#               'fiberMag_i_max_fiberID', 'fiberMag_i_min_fiberID', 'fiberMag_i_mean_fiberID', 'fiberMag_i_median_fiberID', 'fiberMag_i_std_fiberID',

#               'fiberMag_z_max_fiberID', 'fiberMag_z_min_fiberID', 'fiberMag_z_mean_fiberID', 'fiberMag_z_median_fiberID', 'fiberMag_z_std_fiberID',

#               'fiberMag_mean_max_fiberID', 'fiberMag_mean_min_fiberID', 'fiberMag_mean_mean_fiberID', 'fiberMag_mean_median_fiberID', 'fiberMag_mean_std_fiberID',

#               'fiberMag_median_max_fiberID', 'fiberMag_median_min_fiberID', 'fiberMag_median_mean_fiberID', 'fiberMag_median_median_fiberID', 'fiberMag_median_std_fiberID',

#               'fiberMag_std_max_fiberID', 'fiberMag_std_min_fiberID', 'fiberMag_std_mean_fiberID', 'fiberMag_std_median_fiberID', 'fiberMag_std_std_fiberID',

#               'fiberMag_sum_max_fiberID', 'fiberMag_sum_min_fiberID', 'fiberMag_sum_mean_fiberID', 'fiberMag_sum_median_fiberID', 'fiberMag_sum_std_fiberID'

#               ]

# gb.head(2)



# train = pd.merge(train, gb, how='left', on='fiberID')

# train.head(2)
# did not use these features.

# # Statistics and distance based features

# gb2 = test.groupby('fiberID', as_index=False).agg({'fiberMag_u': {'fiberMag_u_max_fiberID': np.max, 'fiberMag_u_min_fiberID': np.min, 

#                                                                   'fiberMag_u_mean_fiberID' : np.mean, 'fiberMag_u_median_fiberID' : np.median,

#                                                                   'fiberMag_u_std_fiberID' : np.std},

#                                                    'fiberMag_g': {'fiberMag_g_max_fiberID': np.max, 'fiberMag_g_min_fiberID': np.min, 

#                                                                   'fiberMag_g_mean_fiberID' : np.mean, 'fiberMag_g_median_fiberID' : np.median,

#                                                                   'fiberMag_g_std_fiberID' : np.std},

#                                                    'fiberMag_r': {'fiberMag_r_max_fiberID': np.max, 'fiberMag_r_min_fiberID': np.min, 

#                                                                   'fiberMag_r_mean_fiberID' : np.mean, 'fiberMag_r_median_fiberID' : np.median,

#                                                                   'fiberMag_r_std_fiberID' : np.std},

#                                                    'fiberMag_i': {'fiberMag_i_max_fiberID': np.max, 'fiberMag_i_min_fiberID': np.min, 

#                                                                   'fiberMag_i_mean_fiberID' : np.mean, 'fiberMag_i_median_fiberID' : np.median,

#                                                                   'fiberMag_i_std_fiberID' : np.std},

#                                                    'fiberMag_z': {'fiberMag_z_max_fiberID': np.max, 'fiberMag_z_min_fiberID': np.min, 

#                                                                   'fiberMag_z_mean_fiberID' : np.mean, 'fiberMag_z_median_fiberID' : np.median,

#                                                                   'fiberMag_z_std_fiberID' : np.std},

#                                                    'fiberMag_mean' : {'fiberMag_mean_max_fiberID': np.max, 'fiberMag_mean_min_fiberID': np.min, 

#                                                                   'fiberMag_mean_mean_fiberID' : np.mean, 'fiberMag_mean_median_fiberID' : np.median,

#                                                                   'fiberMag_mean_std_fiberID' : np.std},

#                                                    'fiberMag_median' : {'fiberMag_median_max_fiberID': np.max, 'fiberMag_median_min_fiberID': np.min, 

#                                                                   'fiberMag_median_mean_fiberID' : np.mean, 'fiberMag_median_median_fiberID' : np.median,

#                                                                   'fiberMag_median_std_fiberID' : np.std},

#                                                    'fiberMag_std' : {'fiberMag_std_max_fiberID': np.max, 'fiberMag_std_min_fiberID': np.min, 

#                                                                   'fiberMag_std_mean_fiberID' : np.mean, 'fiberMag_std_median_fiberID' : np.median,

#                                                                   'fiberMag_std_std_fiberID' : np.std},

#                                                    'fiberMag_sum' : {'fiberMag_sum_max_fiberID': np.max, 'fiberMag_sum_min_fiberID': np.min, 

#                                                                   'fiberMag_sum_mean_fiberID' : np.mean, 'fiberMag_sum_median_fiberID' : np.median,

#                                                                   'fiberMag_sum_std_fiberID' : np.std}

#                                                    }).fillna(0)



# gb2.columns = ['fiberID', 'fiberMag_u_max_fiberID', 'fiberMag_u_min_fiberID', 'fiberMag_u_mean_fiberID', 'fiberMag_u_median_fiberID', 'fiberMag_u_std_fiberID',

#               'fiberMag_g_max_fiberID', 'fiberMag_g_min_fiberID', 'fiberMag_g_mean_fiberID', 'fiberMag_g_median_fiberID', 'fiberMag_g_std_fiberID',

#               'fiberMag_r_max_fiberID', 'fiberMag_r_min_fiberID', 'fiberMag_r_mean_fiberID', 'fiberMag_r_median_fiberID', 'fiberMag_r_std_fiberID',

#               'fiberMag_i_max_fiberID', 'fiberMag_i_min_fiberID', 'fiberMag_i_mean_fiberID', 'fiberMag_i_median_fiberID', 'fiberMag_i_std_fiberID',

#               'fiberMag_z_max_fiberID', 'fiberMag_z_min_fiberID', 'fiberMag_z_mean_fiberID', 'fiberMag_z_median_fiberID', 'fiberMag_z_std_fiberID',

#               'fiberMag_mean_max_fiberID', 'fiberMag_mean_min_fiberID', 'fiberMag_mean_mean_fiberID', 'fiberMag_mean_median_fiberID', 'fiberMag_mean_std_fiberID',

#               'fiberMag_median_max_fiberID', 'fiberMag_median_min_fiberID', 'fiberMag_median_mean_fiberID', 'fiberMag_median_median_fiberID', 'fiberMag_median_std_fiberID',

#               'fiberMag_std_max_fiberID', 'fiberMag_std_min_fiberID', 'fiberMag_std_mean_fiberID', 'fiberMag_std_median_fiberID', 'fiberMag_std_std_fiberID',

#               'fiberMag_sum_max_fiberID', 'fiberMag_sum_min_fiberID', 'fiberMag_sum_mean_fiberID', 'fiberMag_sum_median_fiberID', 'fiberMag_sum_std_fiberID'

#               ]



# gb2.isnull().sum()

# test = pd.merge(test, gb2, how='left', on='fiberID')

# print(train.isnull().sum().sum())

# print(test.isnull().sum().sum())

# test.head(2)
train_gb = train.copy()

X_train_gb = train_gb.drop('type', axis=1)

y_train_gb = train_gb['type']

test_gb = test.copy()
unique_labels = train['type'].unique()

label_dict = {val: i for i, val in enumerate(unique_labels)}

i2lb = {v:k for k, v in label_dict.items()}

i2lb
scaler = StandardScaler()

labels = train['type']

train = train.drop('type', axis=1)

_mat = scaler.fit_transform(train)

train = pd.DataFrame(_mat, columns=train.columns, index=train.index)

train_x = train

train_y = labels.replace(label_dict)

test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

test.head(2)
test_ids = test.index.copy()

test_ids = pd.DataFrame(test_ids, index=test_ids)

test_ids
# # data set split

# X_train, X_vali, y_train, y_vali = train_test_split(X_train_gb, y_train_gb, test_size=0.2,

#                                                                     stratify=y_train_gb, random_state=5)
def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod(

            (datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' %

              (thour, tmin, round(tsec, 2)))
# XGboost Cross Validation



folds = 5



xgb_cv_sum = 0

xgb_pred = []

xgb_fpred = []



avreal = y_train_gb



# blend_train = []

# blend_test = []



train_time = timer(None)

kf = StratifiedKFold(n_splits=folds, random_state=5, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(X_train_gb, y_train_gb)):

    start_time = timer(None)

    Xtrain, Xval = X_train_gb.iloc[train_index], X_train_gb.iloc[val_index]

    ytrain, yval = y_train_gb.iloc[train_index], y_train_gb.iloc[val_index]



    model = XGBClassifier(random_state=5,

                         max_depth=10,

                         n_estimators=3500,

                         learning_rate=0.005,

                         objective='multi:softprob',

                         tree_method='gpu_hist',

                         colsample_bytree=0.5,

                         subsample=0.6

                          )

    

    model.fit(Xtrain, ytrain, eval_set=[(Xval, yval)],

              eval_metric='mlogloss', early_stopping_rounds=100, verbose=1000)

              

    xgb_scores_val = model.predict_proba(Xval)

    xgb_log_loss = log_loss(yval, xgb_scores_val)

    print('\n Fold %02d xgb mlogloss: %.6f' % ((i + 1), xgb_log_loss))

    xgb_y_pred = model.predict_proba(test_gb)



    del Xtrain, Xval

    gc.collect()



    timer(start_time)



    if i > 0:

        xgb_fpred = xgb_pred + xgb_y_pred

    else:

        xgb_fpred = xgb_y_pred

    xgb_pred = xgb_fpred

    xgb_cv_sum = xgb_cv_sum + xgb_log_loss



timer(train_time)



xgb_cv_score = (xgb_cv_sum / folds)



print('\n Average xgb mlogloss:\t%.6f' % xgb_cv_score)

xgb_score = round(xgb_cv_score, 6)



xgb_mpred = xgb_pred / folds
# LGBM Cross Validation



folds = 5



lgb_cv_sum = 0

lgb_pred = []

lgb_fpred = []



avreal = y_train_gb



# blend_train = []

# blend_test = []



train_time = timer(None)

kf = StratifiedKFold(n_splits=folds, random_state=5, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(X_train_gb, y_train_gb)):

    start_time = timer(None)

    Xtrain, Xval = X_train_gb.iloc[train_index], X_train_gb.iloc[val_index]

    ytrain, yval = y_train_gb.iloc[train_index], y_train_gb.iloc[val_index]



    model = LGBMClassifier(random_state=15,

                           num_leaves=140,

                           n_estimators=2000,

                           learning_rate=0.01,

                           boost_from_average=True,

                           colsample_bytree=0.4,

                           subsample=0.6,

                           min_child_samples=200,

                           objective='multiclass'

    )

    model.fit(Xtrain, ytrain, eval_set=(Xval, yval),

              early_stopping_rounds=100, verbose=500)

              

    lgb_scores_val = model.predict_proba(Xval)

    lgb_log_loss = log_loss(yval, lgb_scores_val)

    print('\n Fold %02d lgb mlogloss: %.6f' % ((i + 1), lgb_log_loss))

    lgb_y_pred = model.predict_proba(test_gb)



    del Xtrain, Xval

    gc.collect()



    timer(start_time)



    if i > 0:

        lgb_fpred = lgb_pred + lgb_y_pred

    else:

        lgb_fpred = lgb_y_pred

    lgb_pred = lgb_fpred

    lgb_cv_sum = lgb_cv_sum + lgb_log_loss



timer(train_time)



lgb_cv_score = (lgb_cv_sum / folds)



print('\n Average lgb mlogloss:\t%.6f' % lgb_cv_score)

lgb_score = round(lgb_cv_score, 6)



lgb_mpred = lgb_pred / folds
from keras.callbacks import LearningRateScheduler

import math

# learning rate schedule

def step_decay(epoch):

	initial_lrate = 0.002

	drop = 0.1

	epochs_drop = 30

	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))

	return lrate



# learning schedule callback

lrate = LearningRateScheduler(step_decay)
es = EarlyStopping(monitor='val_loss',patience=30)
opti = keras.optimizers.Adam(lr = 0.002)
def build_model():

    model = Sequential()

    model.add(Dense(units=1024, input_dim=len(train_x.columns)))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=19, activation='softmax'))



    model.compile(optimizer=opti,

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    return model
def build_model_2():

    model = Sequential()

    model.add(Dense(units=1024, input_dim=len(train_x.columns)))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=2048))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dropout(0.05))

    model.add(Dense(units=3000))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dropout(0.1))

    model.add(Dense(units=3000))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dropout(0.1))

    model.add(Dense(units=2048))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dropout(0.05))

    model.add(Dense(units=1024))

    model.add(BatchNormalization())

    model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(units=512))

    model.add(BatchNormalization())

    model.add(Dense(units=19, activation='softmax'))



    model.compile(optimizer=opti,

                  loss='sparse_categorical_crossentropy',

                  metrics=['accuracy'])

    return model
# Keras NN model_1 CV

folds = 10



MLP_cv_sum = 0

MLP_pred = []

MLP_fpred = []



avreal = train_y



# blend_train = []

# blend_test = []



train_time = timer(None)

kf = StratifiedKFold(n_splits=folds, random_state=5, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(train_x, train_y)):

    start_time = timer(None)

    Xtrain, Xval = train_x.iloc[train_index], train_x.iloc[val_index]

    ytrain, yval = train_y.iloc[train_index], train_y.iloc[val_index]



    model = build_model()

    

    path = "fold " + str(i+1) + " bestmodel.hdf5"

    check_best_model = keras.callbacks.ModelCheckpoint(filepath=path, monitor='val_loss',

                                                   verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    callback_list = [lrate, es, check_best_model]

    

    model.fit(Xtrain, ytrain,

              validation_data=(Xval, yval),

              batch_size=512,

              epochs=150,

              callbacks=callback_list,

              verbose=0

              )

    

    # best model load

    model = load_model(path)

    

    MLP_scores_val = model.predict(Xval)

    MLP_log_loss = log_loss(yval, MLP_scores_val)

    print('\n Fold %02d MLP mlogloss: %.6f' % ((i + 1), MLP_log_loss))

    MLP_y_pred = model.predict(test)

    



    del Xtrain, Xval

    gc.collect()



    timer(start_time)



    if i > 0:

        MLP_fpred = MLP_pred + MLP_y_pred

    else:

        MLP_fpred = MLP_y_pred

    MLP_pred = MLP_fpred

    MLP_cv_sum = MLP_cv_sum + MLP_log_loss



timer(train_time)



MLP_cv_score = (MLP_cv_sum / folds)



print('\n Average MLP mlogloss:\t%.6f' % MLP_cv_score)

MLP_score = round(MLP_cv_score, 6)



MLP_mpred_1 = MLP_pred / folds
# Keras NN model_2 CV

folds = 10



MLP_cv_sum = 0

MLP_pred = []

MLP_fpred = []



avreal = train_y



# blend_train = []

# blend_test = []



train_time = timer(None)

kf = StratifiedKFold(n_splits=folds, random_state=5, shuffle=True)

for i, (train_index, val_index) in enumerate(kf.split(train_x, train_y)):

    start_time = timer(None)

    Xtrain, Xval = train_x.iloc[train_index], train_x.iloc[val_index]

    ytrain, yval = train_y.iloc[train_index], train_y.iloc[val_index]



    model = build_model_2()

    

    path = "fold " + str(i+1) + " bestmodel_2.hdf5"

    check_best_model = keras.callbacks.ModelCheckpoint(filepath=path, monitor='val_loss',

                                                   verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    callback_list = [lrate, es, check_best_model]

    

    model.fit(Xtrain, ytrain,

              validation_data=(Xval, yval),

              batch_size=512,

              epochs=150,

              callbacks=callback_list,

              verbose=0

              )

    

    # best model load

    model = load_model(path)

    

    MLP_scores_val = model.predict(Xval)

    MLP_log_loss = log_loss(yval, MLP_scores_val)

    print('\n Fold %02d MLP mlogloss: %.6f' % ((i + 1), MLP_log_loss))

    MLP_y_pred = model.predict(test)

    



    del Xtrain, Xval

    gc.collect()



    timer(start_time)



    if i > 0:

        MLP_fpred = MLP_pred + MLP_y_pred

    else:

        MLP_fpred = MLP_y_pred

    MLP_pred = MLP_fpred

    MLP_cv_sum = MLP_cv_sum + MLP_log_loss



timer(train_time)



MLP_cv_score = (MLP_cv_sum / folds)



print('\n Average MLP mlogloss:\t%.6f' % MLP_cv_score)

MLP_score = round(MLP_cv_score, 6)



MLP_mpred_2 = MLP_pred / folds
xgb = XGBClassifier(n_estimators=1)

xgb.fit(X_train_gb, y_train_gb)

print(xgb.classes_)
lgb = LGBMClassifier(n_estimators=1)

lgb.fit(X_train_gb, y_train_gb)

print(lgb.classes_)
sample = pd.read_csv('/kaggle/input/dacon-stage-2/sample_submission.csv')

sample.head(2)
xgb_mat = pd.DataFrame(xgb_mpred, index=test_gb.index, columns=xgb.classes_)

xgb_mat = pd.concat([test_ids, xgb_mat], axis=1)

xgb_mat = xgb_mat[sample.columns]

xgb_mat = xgb_mat.drop('id', axis=1)

xgb_mat.head(2)
lgb_mat = pd.DataFrame(lgb_mpred, index=test_gb.index, columns=lgb.classes_)

lgb_mat = pd.concat([test_ids, lgb_mat], axis=1)

lgb_mat = lgb_mat[sample.columns]

lgb_mat = lgb_mat.drop('id', axis=1)

lgb_mat.head(2)
MLP_mat_1 = pd.DataFrame(MLP_mpred_1, index=test.index)

MLP_mat_1 = MLP_mat_1.rename(columns=i2lb)

MLP_mat_1 = pd.concat([test_ids, MLP_mat_1], axis=1)

MLP_mat_1 = MLP_mat_1[sample.columns]

MLP_mat_1 = MLP_mat_1.drop('id', axis=1)

MLP_mat_1.head(2)
MLP_mat_2 = pd.DataFrame(MLP_mpred_2, index=test.index)

MLP_mat_2 = MLP_mat_2.rename(columns=i2lb)

MLP_mat_2 = pd.concat([test_ids, MLP_mat_2], axis=1)

MLP_mat_2 = MLP_mat_2[sample.columns]

MLP_mat_2 = MLP_mat_2.drop('id', axis=1)

MLP_mat_2.head(2)
submission = 0.4*MLP_mat_1 + 0.3*MLP_mat_2 + 0.2*xgb_mat + 0.1*lgb_mat

submission = pd.concat([test_ids, submission], axis=1)

submission.head()
submission.to_csv('dacon_stage_2_submission_model_ensemble.csv', index=False)