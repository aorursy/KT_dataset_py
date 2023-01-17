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



import plotly.graph_objects as go

from plotly.subplots import make_subplots



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



pd.set_option("display.precision", 8)



DIR_INPUT = '/kaggle/input/liverpool-ion-switching'
train_df = pd.read_csv(DIR_INPUT + '/train.csv')

train_df.shape
train_df.head()
train_df.info()
train_df.describe()
train_df['time'].diff().min(), train_df['time'].diff().max()
train_df['open_channels'].value_counts()
train_df.iloc[0:500000]['open_channels'].value_counts()
train_df['signal'].min(), train_df['signal'].max(), train_df['signal'].mean()
fig = go.Figure(data=[

    go.Scatter(x=train_df.iloc[100000:125000]['time'], y=train_df.iloc[100000:125000]['signal'], name='Signal'),

])



fig.update_layout(title='Signal (part of batch #0)')

fig.show()
batch = train_df.iloc[2200000:2202000]

batch.reset_index(drop=True, inplace=True)



data=[

    go.Scatter(x=batch.index, y=batch['signal'], name='Signal'),

]



for i in range(11):

    ocx = batch[batch['open_channels'] == i]

    data.append(go.Scatter(x=ocx.index, y=[i for _ in range(len(ocx))], name='OC: {}'.format(i), marker=dict(size=4), mode="markers"))



fig = go.Figure(data)



fig.update_layout(title='Signal (part of batch #4)')

fig.show()
fig = go.Figure(data=[

    go.Bar(x=list(range(11)), y=train_df['open_channels'].value_counts().values)

])



fig.update_layout(title='Target (open_channels) distribution')

fig.show()
fig = make_subplots(rows=3, cols=4,  subplot_titles=["Batch #{}".format(i) for i in range(10)])

i = 0

for row in range(1, 4):

    for col in range(1, 5):

        data = train_df.iloc[(i * 500000):((i+1) * 500000 + 1)]['open_channels'].value_counts().values

        fig.add_trace(go.Bar(x=list(range(11)), y=data), row=row, col=col)

        

        i += 1





fig.update_layout(title_text="Target distribution in different batches", showlegend=False)

fig.show()
window_sizes = [10, 50, 100, 1000]



for window in window_sizes:

    train_df["rolling_mean_" + str(window)] = train_df['signal'].rolling(window=window).mean()

    train_df["rolling_std_" + str(window)] = train_df['signal'].rolling(window=window).std()
fig, ax = plt.subplots(len(window_sizes),1,figsize=(20, 6 * len(window_sizes)))



n = 0

for col in train_df.columns.values:

    if "rolling_" in col:

        if "mean" in col:

            mean_df = train_df.iloc[2200000:2210000][col]

            ax[n].plot(mean_df, label=col, color="mediumseagreen")

        if "std" in col:

            std = train_df.iloc[2200000:2210000][col].values

            ax[n].fill_between(mean_df.index.values,

                               mean_df.values-std, mean_df.values+std,

                               facecolor='lightgreen',

                               alpha = 0.5, label=col)

            ax[n].legend()

            n+=1
train_df = pd.read_csv(DIR_INPUT + '/train.csv')

train_df.shape
window_sizes = [50, 100, 1000, 5000, 10000, 25000]



for window in window_sizes:

    train_df["rolling_mean_" + str(window)] = train_df['signal'].rolling(window=window).mean()

    train_df["rolling_std_" + str(window)] = train_df['signal'].rolling(window=window).std()

    train_df["rolling_min_" + str(window)] = train_df['signal'].rolling(window=window).min()

    train_df["rolling_max_" + str(window)] = train_df['signal'].rolling(window=window).max()

    

    

    a = (train_df['signal'] - train_df['rolling_min_' + str(window)]) / (train_df['rolling_max_' + str(window)] - train_df['rolling_min_' + str(window)])

    train_df["norm_" + str(window)] = a * (np.floor(train_df['rolling_max_' + str(window)]) - np.ceil(train_df['rolling_min_' + str(window)]))

    

train_df.fillna(0, inplace=True)



train_y = train_df['open_channels']

train_x = train_df.drop(columns=['time', 'open_channels'])



del train_df
scaler = StandardScaler()

scaler.fit(train_x)

train_x_scaled = pd.DataFrame(scaler.transform(train_x), columns=train_x.columns)



del train_x
test_df = pd.read_csv(DIR_INPUT + '/test.csv')

test_df.drop(columns=['time'], inplace=True)

test_df.shape
for window in window_sizes:

    test_df["rolling_mean_" + str(window)] = test_df['signal'].rolling(window=window).mean()

    test_df["rolling_std_" + str(window)] = test_df['signal'].rolling(window=window).std()

    test_df["rolling_min_" + str(window)] = test_df['signal'].rolling(window=window).min()

    test_df["rolling_max_" + str(window)] = test_df['signal'].rolling(window=window).max()

    

    

    a = (test_df['signal'] - test_df['rolling_min_' + str(window)]) / (test_df['rolling_max_' + str(window)] - test_df['rolling_min_' + str(window)])

    test_df["norm_" + str(window)] = a * (np.floor(test_df['rolling_max_' + str(window)]) - np.ceil(test_df['rolling_min_' + str(window)]))

    

test_df.fillna(0, inplace=True)

test_x_scaled = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)

del test_df
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



params = {'num_leaves': 128,

          'min_data_in_leaf': 64,

          'objective': 'huber',

          'max_depth': -1,

          'learning_rate': 0.01,

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