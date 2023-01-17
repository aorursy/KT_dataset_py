import numpy as np 

import pandas as pd

from sklearn import *

import lightgbm as lgb

import xgboost as xgb

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy as sp

sns.set(style="whitegrid")
PLOT_ALL = True
df_train = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

df_test = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
df = pd.concat([df_train.drop(['open_channels'], axis=1), df_test])

df_train.shape, df_test.shape, df.shape
batch_time = 500000 # The signal batch time period
df_train.describe()
df_test.describe()
df.describe()
oc_vc = df_train['open_channels'].value_counts()

ax = sns.barplot(x=oc_vc.index, y=oc_vc.values)
if PLOT_ALL:

    plt.figure(figsize=(12,12))

    for i in range(10):

        plt.subplot(5,2,i+1)

        idx = range(batch_time*i, batch_time*(i+1)-1)

        oc_vc = df_train.loc[idx, 'open_channels'].value_counts()

        ax = sns.barplot(x=oc_vc.index, y=oc_vc.values)
if PLOT_ALL:

    plt.figure(figsize=(12,12))

    for i in range(10):

        plt.subplot(5,2,i+1)

        idx = range(batch_time*i, batch_time*(i+1)-1)

        ax = sns.lineplot(x="time", y="open_channels", data=df_train[batch_time*i:batch_time*(i+1)-1])
idx = 0

plt.figure(figsize=(18,24))

for i, d in df_train.groupby('open_channels'):

    plt.subplot(4,3,idx+1)

    sns.distplot(d['signal'], bins=50)

    plt.title('Signal Distribution for %d number of open channels' % i)

    idx += 1
if PLOT_ALL:

    sns.pairplot(x_vars=["signal"], y_vars=["open_channels"], data=df_train, size=6)

    plt.title('Signal Distribution for different number of open channels')
non_zero_chn_index = df_train.index[df_train['open_channels'] > 0]

print(non_zero_chn_index.shape)



def plot_signal(mid_idx, plot_len):

    plt.figure(figsize=(12,12))

    for i in range(1,3):

        start = mid_idx[i-1] - plot_len

        end = mid_idx[i-1] + plot_len



        plt.subplot(2,2,i)

        plt.title('Open Channels, Time: %d - %d' % (start, end))

        sns.lineplot(df_train.loc[start:end, 'time'], df_train.loc[start:end, 'open_channels'])



        plt.subplot(2,2,i+2)

        plt.title('Signal, Time: %d - %d' % (start, end))

        sns.lineplot(df_train.loc[start:end, 'time'], df_train.loc[start:end, 'signal'])
if PLOT_ALL:

    mid_idx = [non_zero_chn_index[100000], non_zero_chn_index[1759848]]

    plot_len = 1000

    plot_signal(mid_idx, plot_len)
if PLOT_ALL:

    mid_idx = [non_zero_chn_index[10000], non_zero_chn_index[1759848]]

    plot_len = 20

    plot_signal(mid_idx, plot_len)
if PLOT_ALL:

    plt.figure(figsize=(6,6))

    sns.distplot(df_train['signal'], bins=20)

    sns.distplot(df_test['signal'], bins=20)

    plt.title('Signal Distribution for Test and Train')

    plt.legend(labels=['Train', 'Test'])
plt.figure(figsize=(18,24))

for i in range(12):

    ax = plt.subplot(4,3,i+1)

    batch_idx = 2

    pd.plotting.lag_plot(df_train['open_channels'][batch_time*batch_idx:batch_time*(batch_idx+1)-1], lag=i+1, ax=ax)

    plt.title('Signal lag plot for batch %d' % (i+1))
plt.figure(figsize=(18,24))

for i in range(10):

    ax = plt.subplot(4,3,i+1)

    pd.plotting.autocorrelation_plot(df['signal'][batch_time*i:batch_time*(i+1)-1], ax=ax)

    plt.ylim([-0.25, 0.25])

    plt.title('Signal Autocorrelation for batch %d' % (i+1))
f_s = 10000



plt.figure(figsize=(18,18))

for i in range(10):

    plt.subplot(5,2,i+1)

    x = df['signal'][batch_time*i:batch_time*(i+1)-1]

    X = np.fft.fft(x)

    freqs = np.fft.fftfreq(len(x)) * f_s



    start = 0

    end = len(x) // 100

    plt.plot(freqs[start:end], np.abs(X)[start:end])

    plt.title("Frequency spectrum of the signal, Batch %d" % (i+1))
def add_window_feature(df):

    window_sizes = [10, 25, 50, 100, 500, 1000, 5000, 10000, 25000]

    for window in window_sizes:

        df["rolling_mean_" + str(window)] = df['signal'].rolling(window=window).mean()

        df["rolling_std_" + str(window)] = df['signal'].rolling(window=window).std()

        df["rolling_var_" + str(window)] = df['signal'].rolling(window=window).var()

        df["rolling_min_" + str(window)] = df['signal'].rolling(window=window).min()

        df["rolling_max_" + str(window)] = df['signal'].rolling(window=window).max()

        

    df = df.replace([np.inf, -np.inf], np.nan)

    df.fillna(0, inplace=True)

    

    return df
df_train = add_window_feature(df_train)

print(df_train.columns)

df_test = add_window_feature(df_test)

print(df_test.columns)
X = df_train.drop(['time', 'open_channels'], axis=1)

y = df_train['open_channels']
model = xgb.XGBRegressor(max_depth=3)

model.fit(X,y)
model = lgbm.LGBMRegressor(n_estimators=100)

model.fit(X, y)
X_test = df_test.drop(['time'], axis=1)

preds = model.predict(X_test)
df_test['open_channels'] = np.round(np.clip(preds, 0, 10)).astype(int)

df_test[['time','open_channels']].to_csv('submission.csv', index=False, float_format='%.4f')