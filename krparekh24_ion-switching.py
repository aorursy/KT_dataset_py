# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

import pandas as pd

import numpy as np



import lightgbm as lgb

import time

import datetime



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import LinearRegression



import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/liverpool-ion-switching/train.csv")

test=pd.read_csv("/kaggle/input/liverpool-ion-switching/test.csv")

train.shape
test.shape
train.head()
train['time'].diff().max(),   train['time'].diff().min()
train['open_channels'].value_counts()
train.iloc[0:500000]['open_channels'].value_counts()
train['signal'].min(),train['signal'].max(),train['signal'].mean()
import plotly.graph_objects as go



fig = go.Figure(data=[

    go.Scatter(x=train.iloc[100000:]['time'], y=train.iloc[100000:]['signal'], name='Signal'),])



fig.update_layout(title='Signal (part of batch #0)')

fig.show()
fig = go.Figure(data=[

    go.Scatter(x=train.iloc[100000:125000]['time'], y=train.iloc[100000:125000]['signal'], name='Signal'),])



fig.update_layout(title='Signal (part of batch #0)')

fig.show()
fig = go.Figure(data=[

    go.Bar(x=list(range(11)), y=train['open_channels'].value_counts(sort=False).values)

])



fig.update_layout(title='Target (open_channels) distribution')

fig.show()
from plotly.subplots import make_subplots



fig = make_subplots(rows=3, cols=4,  subplot_titles=["Batch #{}".format(i) for i in range(10)])

i = 0

for row in range(1, 4):

    for col in range(1, 5):

        data = train.iloc[(i * 500000):((i+1) * 500000 + 1)]['open_channels'].value_counts(sort=False).values

        fig.add_trace(go.Bar(x=list(range(11)), y=data), row=row, col=col)

        

        i += 1





fig.update_layout(title_text="Target distribution in different batches", showlegend=False)

fig.show()







    
#Rolling features¶



import matplotlib.pyplot as plt



window_sizes = [10, 50, 100, 1000]



for window in window_sizes:

    train["rolling_mean_" + str(window)] = train['signal'].rolling(window=window).mean()

    train["rolling_std_" + str(window)] = train['signal'].rolling(window=window).std()



fig, ax = plt.subplots(len(window_sizes),1,figsize=(20, 6 * len(window_sizes)))



n = 0

for col in train.columns.values:

    if "rolling_" in col:

        if "mean" in col:

            mean_df = train.iloc[2200000:2210000][col]

            ax[n].plot(mean_df, label=col, color="mediumseagreen")

        if "std" in col:

            std = train.iloc[2200000:2210000][col].values

            ax[n].fill_between(mean_df.index.values,

                               mean_df.values-std, mean_df.values+std,

                               facecolor='lightgreen',

                               alpha = 0.5, label=col)

            ax[n].legend()

            n+=1



DIR_INPUT = '/kaggle/input/liverpool-ion-switching'



train_df = pd.read_csv(DIR_INPUT + '/train.csv')

train_df.shape







window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]



for window in window_sizes:

    train_df["rolling_mean_" + str(window)] = train_df['signal'].rolling(window=window).mean()

    train_df["rolling_std_" + str(window)] = train_df['signal'].rolling(window=window).std()

    train_df["rolling_var_" + str(window)] = train_df['signal'].rolling(window=window).var()

    train_df["rolling_min_" + str(window)] = train_df['signal'].rolling(window=window).min()

    train_df["rolling_max_" + str(window)] = train_df['signal'].rolling(window=window).max()

    

    train_df["rolling_min_max_ratio_" + str(window)] = train_df["rolling_min_" + str(window)] / train_df["rolling_max_" + str(window)]

    train_df["rolling_min_max_diff_" + str(window)] = train_df["rolling_max_" + str(window)] - train_df["rolling_min_" + str(window)]

    

    a = (train_df['signal'] - train_df['rolling_min_' + str(window)]) / (train_df['rolling_max_' + str(window)] - train_df['rolling_min_' + str(window)])

    train_df["norm_" + str(window)] = a * (np.floor(train_df['rolling_max_' + str(window)]) - np.ceil(train_df['rolling_min_' + str(window)]))

    

train_df = train_df.replace([np.inf, -np.inf], np.nan)

train_df.fillna(0, inplace=True)



train_y = train_df['open_channels']

train_x = train_df.drop(columns=['time', 'open_channels'])



del train_df



scaler = StandardScaler()

scaler.fit(train_x)

train_x_scaled = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)



del train_x

test = pd.read_csv(DIR_INPUT + '/test.csv')

test.drop(columns=['time'], inplace=True)

test.shape
for window in window_sizes:

    test["rolling_mean_" + str(window)] = test['signal'].rolling(window=window).mean()

    test["rolling_std_" + str(window)] = test['signal'].rolling(window=window).std()

    test["rolling_var_" + str(window)] = test['signal'].rolling(window=window).var()

    test["rolling_min_" + str(window)] = test['signal'].rolling(window=window).min()

    test["rolling_max_" + str(window)] = test['signal'].rolling(window=window).max()

    

    test["rolling_min_max_ratio_" + str(window)] = test["rolling_min_" + str(window)] / test["rolling_max_" + str(window)]

    test["rolling_min_max_diff_" + str(window)] = test["rolling_max_" + str(window)] - test["rolling_min_" + str(window)]



    

    a = (test['signal'] - test['rolling_min_' + str(window)]) / (test['rolling_max_' + str(window)] - test['rolling_min_' + str(window)])

    test["norm_" + str(window)] = a * (np.floor(test['rolling_max_' + str(window)]) - np.ceil(test['rolling_min_' + str(window)]))



test = test.replace([np.inf, -np.inf], np.nan)

test.fillna(0, inplace=True)
test_x_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns)

del test







n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



params = {'num_leaves': 128,

          'min_data_in_leaf': 64,

          'objective': 'huber',

          'max_depth': -1,

          'learning_rate': 0.005,

          "boosting": "gbdt",

          "bagging_freq": 5,

          "bagging_fraction": 0.8,

          "bagging_seed": 11,

          "metric": 'mae',

          "verbosity": -1,

          'reg_alpha': 0.1,

          'reg_lambda': 0.3

         }







oof = np.zeros(len(train_x_scaled))

prediction = np.zeros(len(test_x_scaled))

scores = []



for fold_n, (train_index, valid_index) in enumerate(folds.split(train_x_scaled)):

    print('Fold', fold_n, 'started at', time.ctime())

    X_train, X_valid = train_x_scaled.iloc[train_index], train_x_scaled.iloc[valid_index]

    y_train, y_valid = train_y.iloc[train_index], train_y.iloc[valid_index]

    

    model = lgb.LGBMRegressor(**params, n_estimators = 5000, n_jobs = -1)

    model.fit(X_train, y_train, 

            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',

            verbose=500, early_stopping_rounds=200)



    y_pred_valid = model.predict(X_valid)

    y_pred = model.predict(test_x_scaled, num_iteration=model.best_iteration_)



    oof[valid_index] = y_pred_valid.reshape(-1,)

    scores.append(mean_absolute_error(y_valid, y_pred_valid))



    prediction += y_pred



prediction /= n_fold







sample_df = pd.read_csv(DIR_INPUT + "/sample_submission.csv", dtype={'time':str})



sample_df['open_channels'] = np.round(prediction).astype(np.int)

sample_df.to_csv("submission.csv", index=False, float_format='%.4f')


