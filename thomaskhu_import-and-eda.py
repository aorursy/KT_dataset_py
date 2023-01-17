# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', nrows = 100000, dtype={'acoustic_data':np.int16, 'time_to_failure': np.float64})

train.head(10)
fig, ax = plt.subplots(1,4,figsize=(13,4))

ax[0].plot(train.time_to_failure.values)

ax[0].set_xlabel('Index'); ax[0].set_ylabel('time to failure')

ax[1].plot(train.acoustic_data.values)

ax[1].set_xlabel('Index'); ax[1].set_ylabel('acoustic data')

ax[2].plot(np.diff(train.time_to_failure.values))

ax[2].set_xlabel('Index'); ax[2].set_ylabel('step of time_to_failure')

ax[3].plot(train.acoustic_data.values, train.time_to_failure.values, 'o', alpha=0.1)

ax[3].set_xlabel('acoustic data'); ax[3].set_ylabel('time to failure')

plt.tight_layout(pad=2)
fig, ax = plt.subplots(1,4,figsize=(13,4))

n = 4096*3

ax[0].plot(train.time_to_failure.values[:n])

ax[0].set_xlabel('Index'); ax[0].set_ylabel('time to failure')

ax[1].plot(train.acoustic_data.values)

ax[1].set_xlabel('Index'); ax[1].set_ylabel('acoustic data')

ax[2].plot(np.diff(train.time_to_failure.values))

ax[2].set_xlabel('Index'); ax[2].set_ylabel('step of time_to_failure')

ax[3].plot(train.acoustic_data.values, train.time_to_failure.values, 'o', alpha=0.1)

ax[3].set_xlabel('acoustic data'); ax[3].set_ylabel('time to failure')

plt.tight_layout(pad=2)
%%time

train = pd.read_csv('../input/train.csv' , dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})



#visualize 1% of samples data, first 100 datapoints

train_ad_sample_df = train['acoustic_data'].values[::100]

train_ttf_sample_df = train['time_to_failure'].values[::100]



#function for plotting based on both features

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)



del train_ad_sample_df

del train_ttf_sample_df
%%time

# Generate Features for the data set

def gen_features(X):

    strain = []

    strain.append(X.mean())

    strain.append(X.std())

    strain.append(X.min())

    strain.append(X.max())

    strain.append(X.kurtosis())

    strain.append(X.skew())

    strain.append(np.quantile(X,0.01))

    strain.append(np.quantile(X,0.05))

    strain.append(np.quantile(X,0.95))

    strain.append(np.quantile(X,0.99))

    strain.append(np.abs(X).max())

    strain.append(np.abs(X).mean())

    strain.append(np.abs(X).std())

    return pd.Series(strain)



train = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})



X_train = pd.DataFrame()

y_train = pd.Series()

for df in train:

    ch = gen_features(df['acoustic_data'])

    X_train = X_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
%%time

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame()



for seg_id in submission.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    ch = gen_features(seg['acoustic_data'])

    X_test = X_test.append(ch, ignore_index=True)
from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR
scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
%%time

train_pool = Pool(X_train, y_train)

cat_model = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')

cat_model.fit(X_train, y_train, silent=True)

y_pred_cat = cat_model.predict(X_test)

y_train_cat = cat_model.predict(X_train)



submission['time_to_failure'] = y_pred_cat

submission.to_csv('submission_cat.csv')
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_train,y_train_cat)
%%time



parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]



reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

reg1.fit(X_train_scaled, y_train.values.flatten())

y_pred1 = reg1.predict(X_train_scaled)



print("Best CV score: {:.4f}".format(reg1.best_score_))

print(reg1.best_params_)



y_pred_SVR = reg1.predict(X_test_scaled)

submission['time_to_failure'] = y_pred_SVR

submission.to_csv('submission_SVR.csv')
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_train,y_pred1)