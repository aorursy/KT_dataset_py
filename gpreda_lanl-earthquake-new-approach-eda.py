import gc

import os

import time

import logging

import datetime

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import xgboost as xgb

import lightgbm as lgb

from scipy import stats

from tqdm import tqdm_notebook

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')
PATH="../input/"

os.listdir(PATH)
print("There are {} files in test folder".format(len(os.listdir(os.path.join(PATH, 'test' )))))
%%time

train_df = pd.read_csv(os.path.join(PATH,'train.csv'), dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))
rows = 150000

segments = int(np.floor(train_df.shape[0] / rows))

print("Number of segments: ", segments)
train_X = pd.DataFrame(index=range(segments), dtype=np.float64)

train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
def create_features(seg_id, seg, X):

    xc = pd.Series(seg['acoustic_data'].values)   

    zc = np.fft.fft(xc)

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    X.loc[seg_id, 'mean'] = xc.mean()

    X.loc[seg_id, 'std'] = xc.std()

    X.loc[seg_id, 'max'] = xc.max()

    X.loc[seg_id, 'min'] = xc.min()

    X.loc[seg_id, 'sum'] = xc.sum()

    X.loc[seg_id, 'mad'] = xc.mad()

    X.loc[seg_id, 'kurt'] = xc.kurtosis()

    X.loc[seg_id, 'skew'] = xc.skew()

    X.loc[seg_id, 'med'] = xc.median()

    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()

    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)

    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)

    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)

    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)

    X.loc[seg_id, 'Rmean'] = realFFT.mean()

    X.loc[seg_id, 'Rstd'] = realFFT.std()

    X.loc[seg_id, 'Rmax'] = realFFT.max()

    X.loc[seg_id, 'Rmin'] = realFFT.min()

    X.loc[seg_id, 'Imean'] = imagFFT.mean()

    X.loc[seg_id, 'Istd'] = imagFFT.std()

    X.loc[seg_id, 'Imax'] = imagFFT.max()

    X.loc[seg_id, 'Imin'] = imagFFT.min()

    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()

    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()

    X.loc[seg_id, 'std_first_25000'] = xc[:25000].std()

    X.loc[seg_id, 'std_last_25000'] = xc[-25000:].std()

    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()

    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()

# iterate over all segments

for seg_id in tqdm_notebook(range(segments)):

    seg = train_df.iloc[seg_id*rows:seg_id*rows+rows]

    create_features(seg_id, seg, train_X)

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
for seg_id in tqdm_notebook(test_X.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    create_features(seg_id, seg, test_X)
print("Train X: {} y: {} Test X: {}".format(train_X.shape, train_y.shape, test_X.shape))
train_X.head()
test_X.head()
def plot_distplot(feature):

    plt.figure(figsize=(16,6))

    plt.title("Distribution of {} values in the train and test set".format(feature))

    sns.distplot(train_X[feature],color="green", kde=True,bins=120, label='train')

    sns.distplot(test_X[feature],color="blue", kde=True,bins=120, label='test')

    plt.legend()

    plt.show()
def plot_distplot_features(features, nlines=4, colors=['green', 'blue'], df1=train_X, df2=test_X):

    i = 0

    plt.figure()

    fig, ax = plt.subplots(nlines,2,figsize=(16,4*nlines))

    for feature in features:

        i += 1

        plt.subplot(nlines,2,i)

        sns.distplot(df1[feature],color=colors[0], kde=True,bins=40, label='train')

        sns.distplot(df2[feature],color=colors[1], kde=True,bins=40, label='test')

    plt.show()
features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurt', 'skew']

plot_distplot_features(features)
features = ['med','abs_mean', 'q95', 'q99', 'q05', 'q01']

plot_distplot_features(features,3)
features = ['Rmean', 'Rstd', 'Rmax','Rmin', 'Imean', 'Istd', 'Imax', 'Imin']

plot_distplot_features(features)
features = ['std_first_50000', 'std_last_50000', 'std_first_25000','std_last_25000', 'std_first_10000','std_last_10000']

plot_distplot_features(features,3)
scaler = StandardScaler()

scaler.fit(pd.concat([train_X, test_X]))

scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurt', 'skew']

plot_distplot_features(features, nlines=4, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)
features = ['med','abs_mean', 'q95', 'q99', 'q05', 'q01']

plot_distplot_features(features, nlines=3, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)
features = ['Rmean', 'Rstd', 'Rmax','Rmin', 'Imean', 'Istd', 'Imax', 'Imin']

plot_distplot_features(features, nlines=4, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)
features = ['std_first_50000', 'std_last_50000', 'std_first_25000','std_last_25000', 'std_first_10000','std_last_10000']

plot_distplot_features(features, nlines=3, colors=['red', 'magenta'], df1=scaled_train_X, df2=scaled_test_X)
def plot_acc_agg_ttf_data(feature, title="Averaged accoustic data and ttf"):

    fig, ax1 = plt.subplots(figsize=(16, 8))

    plt.title('Averaged accoustic data ({}) and time to failure'.format(feature))

    plt.plot(train_X[feature], color='r')

    ax1.set_xlabel('training samples')

    ax1.set_ylabel('acoustic data ({})'.format(feature), color='r')

    plt.legend(['acoustic data ({})'.format(feature)], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_y, color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)
plot_acc_agg_ttf_data('mean')
plot_acc_agg_ttf_data('std')
plot_acc_agg_ttf_data('max')
plot_acc_agg_ttf_data('min')
plot_acc_agg_ttf_data('sum')
plot_acc_agg_ttf_data('mad')
plot_acc_agg_ttf_data('kurt')
plot_acc_agg_ttf_data('skew')
plot_acc_agg_ttf_data('std_first_50000')
plot_acc_agg_ttf_data('std_last_50000')
plot_acc_agg_ttf_data('std_first_25000')
plot_acc_agg_ttf_data('std_last_25000')
plot_acc_agg_ttf_data('std_first_10000')
plot_acc_agg_ttf_data('std_last_10000')