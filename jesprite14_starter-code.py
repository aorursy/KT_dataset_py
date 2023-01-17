# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/X_train.csv')

test = pd.read_csv('../input/X_test.csv')

y = pd.read_csv('../input/y_train.csv')

print(train.head())

print(train.columns)

print(y.head())

print("Length of Train", len(train))

print("Length of Y Labels", len(y))
# Standardize all Columns that are not ID's or measurement numbers

col = train.columns[3:]

scaler = StandardScaler()

# scale the columns that contain the data

new_df = scaler.fit_transform(train[col])

new_df = pd.DataFrame(new_df, columns=col)

# Add back index

new_df["series_id"] = train['series_id']

new_df.head()
np.unique(y['surface'])
y = pd.read_csv('../input/y_train.csv')

y['surface'].value_counts().plot(kind='bar')
le = LabelEncoder()

y = le.fit_transform(y['surface'])

y
def change1(x):

    return np.mean(np.abs(np.diff(x)))



def change2(x):

    return np.mean(np.diff(np.abs(np.diff(x))))



def feature_extraction(raw_frame):

    frame = pd.DataFrame()

    raw_frame['angular_velocity'] = raw_frame['angular_velocity_X'] + raw_frame['angular_velocity_Y'] + raw_frame['angular_velocity_Z']

    raw_frame['linear_acceleration'] = raw_frame['linear_acceleration_X'] + raw_frame['linear_acceleration_Y'] + raw_frame['linear_acceleration_Z']

    raw_frame['velocity_to_acceleration'] = raw_frame['angular_velocity'] / raw_frame['linear_acceleration']

    #raw_frame['acceleration_cumsum'] = raw_frame['linear_acceleration'].cumsum()

    

    for col in raw_frame.columns[3:]:

        frame[col + '_mean'] = raw_frame.groupby(['series_id'])[col].mean()

        frame[col + '_std'] = raw_frame.groupby(['series_id'])[col].std()

        frame[col + '_max'] = raw_frame.groupby(['series_id'])[col].max()

        frame[col + '_min'] = raw_frame.groupby(['series_id'])[col].min()

        frame[col + '_max_to_min'] = frame[col + '_max'] / frame[col + '_min']

        

        # Change 1st order

        frame[col + '_mean_abs_change'] = raw_frame.groupby('series_id')[col].apply(change1)

        # Change 2nd order

        #frame[col + '_mean_abs_change2'] = raw_frame.groupby('series_id')[col].apply(change2)

        frame[col + '_abs_max'] = raw_frame.groupby('series_id')[col].apply(lambda x: np.max(np.abs(x)))

    return frame



train_df = feature_extraction(new_df)

len(train_df)
import lightgbm as lgb

import time

num_folds = 10

target = y



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

    'reg_alpha': 0.2,

    'reg_lambda': 0,

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



print("Timer: {:.1f}s".format(time.time() - t0))
predictions = clf.predict(train_df, parameters = None)
y_pred = np.argmax(predictions, axis = 1)

le.inverse_transform(y_pred)