import gc

import os

import time

import logging

import datetime

import warnings

import numpy as np 

import pandas as pd

import xgboost as xgb

import seaborn as sns

from tqdm import tqdm

import lightgbm as lgb

from scipy import stats

from scipy.signal import hann

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.svm import NuSVR, SVR

from catboost import CatBoostRegressor, Pool

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression

from sklearn.gaussian_process.kernels import ExpSineSquared

from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

warnings.filterwarnings("ignore")



import os

print(os.listdir("../input"))
percent_data = 25

total_data_points = 629145480

train = pd.read_csv('../input/train.csv', nrows=total_data_points*percent_data/100)
test = pd.read_csv('../input/test/seg_004cd2.csv')

print('Size of test data: ', len(test))

print(test.describe())

test.head()
test_files = os.listdir("../input/test")

fig, ax = plt.subplots(4,1, figsize=(20,25))



for n in tqdm(range(4)):

    seg = pd.read_csv("../input/test/" + test_files[n])

    ax[n].plot(seg.acoustic_data.values, c="r")

    ax[n].set_ylabel("Signal")

    ax[n].set_ylim([-300, 300])

    ax[n].set_title("Test - {}".format(test_files[n]));
print(train.dtypes)

pd.options.display.precision = 10

train.head()
train.describe()
train_ad_sample_df = train['acoustic_data'][::50]

train_ttf_sample_df = train['time_to_failure'][::50]



fig, ax1 = plt.subplots(figsize=(12,8))

plt.plot(train_ad_sample_df, color='r')

plt.legend(['acoustic_data'], loc=[0.01, 0.95])

ax2 = ax1.twinx()

plt.plot(train_ttf_sample_df, color='b')

plt.legend(['time_to_failure'], loc=[0.01, 0.9])

plt.grid(True)



del train_ad_sample_df

del train_ttf_sample_df
rows = 150000

segments = int(np.floor(train.shape[0] / rows))

print("Number of segments: ", segments)
train_X = pd.DataFrame(index=range(segments), dtype=np.float64)

train_y = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])
def create_features(seg_id, seg, X):

    

    xc = pd.Series(seg['acoustic_data'].values)

    zc = np.fft.fft(xc)

    

    X.loc[seg_id, 'mean'] = xc.mean()

    X.loc[seg_id, 'std'] = xc.std()

    X.loc[seg_id, 'max'] = xc.max()

    X.loc[seg_id, 'min'] = xc.min()

    

    #FFT transform values

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    

    X.loc[seg_id, 'Rmean'] = realFFT.mean()

    X.loc[seg_id, 'Rstd'] = realFFT.std()

    X.loc[seg_id, 'Rmax'] = realFFT.max()

    X.loc[seg_id, 'Rmin'] = realFFT.min()

    

    X.loc[seg_id, 'Imag_mean'] = imagFFT.mean()

    X.loc[seg_id, 'Imag_std'] = imagFFT.std()

    X.loc[seg_id, 'Imag_max'] = imagFFT.max()

    X.loc[seg_id, 'Imag_min'] = imagFFT.min()

    

    X.loc[seg_id, 'Rmean_last_5000'] = realFFT[-5000:].mean()

    X.loc[seg_id, 'Rstd__last_5000'] = realFFT[-5000:].std()

    X.loc[seg_id, 'Rmax_last_5000'] = realFFT[-5000:].max()

    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()

    X.loc[seg_id, 'Rmean_first_5000'] = realFFT[:5000].mean()

    X.loc[seg_id, 'Rstd__first_5000'] = realFFT[:5000].std()

    X.loc[seg_id, 'Rmax_first_5000'] = realFFT[:5000].max()

    X.loc[seg_id, 'Rmin_first_5000'] = realFFT[:5000].min()

    

    X.loc[seg_id, 'Rmean_last_15000'] = realFFT[-15000:].mean()

    X.loc[seg_id, 'Rstd_last_15000'] = realFFT[-15000:].std()

    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()

    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()

    X.loc[seg_id, 'Rmean_first_15000'] = realFFT[:15000].mean()

    X.loc[seg_id, 'Rstd_first_15000'] = realFFT[:15000].std()

    X.loc[seg_id, 'Rmax_first_15000'] = realFFT[:15000].max()

    X.loc[seg_id, 'Rmin_first_15000'] = realFFT[:15000].min()

    

    X.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(xc))

    X.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(xc) / xc[:-1]))[0])

    X.loc[seg_id, 'abs_max'] = np.abs(xc).max()

    X.loc[seg_id, 'abs_min'] = np.abs(xc).min()

    

    X.loc[seg_id, 'std_first_50000'] = xc[:50000].std()

    X.loc[seg_id, 'std_last_50000'] = xc[-50000:].std()

    

    X.loc[seg_id, 'std_first_10000'] = xc[:10000].std()

    X.loc[seg_id, 'std_last_10000'] = xc[-10000:].std()

    

    X.loc[seg_id, 'avg_first_50000'] = xc[:50000].mean()

    X.loc[seg_id, 'avg_last_50000'] = xc[-50000:].mean()

    

    X.loc[seg_id, 'avg_first_10000'] = xc[:10000].mean()

    X.loc[seg_id, 'avg_last_10000'] = xc[-10000:].mean()

    

    X.loc[seg_id, 'min_first_50000'] = xc[:50000].min()

    X.loc[seg_id, 'min_last_50000'] = xc[-50000:].min()

    

    X.loc[seg_id, 'min_first_10000'] = xc[:10000].min()

    X.loc[seg_id, 'min_last_10000'] = xc[-10000:].min()

    

    X.loc[seg_id, 'max_first_50000'] = xc[:50000].max()

    X.loc[seg_id, 'max_last_50000'] = xc[-50000:].max()

    

    X.loc[seg_id, 'max_first_10000'] = xc[:10000].max()

    X.loc[seg_id, 'max_last_10000'] = xc[-10000:].max()

    

    X.loc[seg_id, 'max_to_min'] = xc.max() / np.abs(xc.min())

    X.loc[seg_id, 'max_to_min_diff'] = xc.max() - np.abs(xc.min())

    X.loc[seg_id, 'count_big'] = len(xc[np.abs(xc) > 500])

    X.loc[seg_id, 'sum'] = xc.sum()

    

    X.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(xc[:50000]) / xc[:50000][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(xc[-50000:]) / xc[-50000:][:-1]))[0])

    

    X.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(xc[:10000]) / xc[:10000][:-1]))[0])

    X.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(xc[-10000:]) / xc[-10000:][:-1]))[0])

    

    X.loc[seg_id, 'q95'] = np.quantile(xc, 0.95)

    X.loc[seg_id, 'q99'] = np.quantile(xc, 0.99)

    X.loc[seg_id, 'q05'] = np.quantile(xc, 0.05)

    X.loc[seg_id, 'q01'] = np.quantile(xc, 0.01)

    

    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)

    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)

    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)

    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)

    

    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()

    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()

    

    X.loc[seg_id, 'mad'] = xc.mad()

    X.loc[seg_id, 'kurt'] = xc.kurtosis()

    X.loc[seg_id, 'skew'] = xc.skew()

    X.loc[seg_id, 'med'] = xc.median()

    

    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()

    X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)

    

    ewma = pd.Series.ewm

    X.loc[seg_id, 'exp_Moving_average_300_mean'] = ewma(xc, span=300).mean().mean(skipna=True)

    X.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(xc, span=3000).mean().mean(skipna=True)

    X.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(xc, span=6000).mean().mean(skipna=True)

    

    no_of_std = 2

    X.loc[seg_id, 'MA_700MA_std_mean'] = xc.rolling(window=700).std().mean()

    X.loc[seg_id,'MA_700MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id,'MA_700MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X.loc[seg_id, 'MA_400MA_std_mean'] = xc.rolling(window=400).std().mean()

    X.loc[seg_id,'MA_400MA_BB_high_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id,'MA_400MA_BB_low_mean'] = (X.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X.loc[seg_id, 'MA_1000MA_std_mean'] = xc.rolling(window=1000).std().mean()

    

    X.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(xc, [75, 25]))

    X.loc[seg_id, 'q999'] = np.quantile(xc,0.999)

    X.loc[seg_id, 'q001'] = np.quantile(xc,0.001)

    X.loc[seg_id, 'ave10'] = stats.trim_mean(xc, 0.1)

    

    for windows in [10, 100, 1000]:

        x_roll_std = xc.rolling(windows).std().dropna().values

        x_roll_mean = xc.rolling(windows).mean().dropna().values

        

        X.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        X.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
for seg_id in tqdm(range(segments)):

    seg = train.iloc[seg_id*rows:seg_id*rows+rows]

    create_features(seg_id, seg, train_X)

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]
print("New training shape: ", train_X.shape)

train_X.head(10)
scaler = StandardScaler()

scaler.fit(train_X)

scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

scaled_train_X.head(10)
#scaler = Normalizer()

#scaler.fit(train_X)

#scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

#scaled_train_X.head(10)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
for seg_id in tqdm(test_X.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    create_features(seg_id, seg, test_X)
scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)
Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

gammas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]

param_grid = {'C': Cs, 'gamma' : gammas}

grid_search = GridSearchCV(SVR(kernel='rbf', tol=0.01), param_grid, cv=5)

grid_search.fit(scaled_train_X, train_y)

print('Best CV Score:', grid_search.best_score_)

print('Best parameters: ', grid_search.best_params_)

print('Best estimator: ', grid_search.best_estimator_)
predictions = grid_search.predict(scaled_test_X)

print(len(predictions))

print(predictions)
param_grid = {"alpha": [1e0, 1e-1, 1e-2, 1e-3],

              "gamma": np.logspace(-2, 2, 5)}

grid_search = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), param_grid, cv=5)

grid_search.fit(scaled_train_X, train_y)

print('Best CV Score:', grid_search.best_score_)

print('Best parameters: ', grid_search.best_params_)

print('Best estimator: ', grid_search.best_estimator_)
predictions = grid_search.predict(scaled_test_X)

print(len(predictions))

print(predictions)
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(scaled_train_X, train_y, train_size=0.75, random_state=42)
model = CatBoostRegressor(iterations=50, depth=3, learning_rate=0.1, loss_function='RMSE')

model.fit(scaled_train_X, train_y, cat_features=None, eval_set=(X_validation, y_validation), plot=True)
predictions = grid_search.predict(scaled_test_X)

print(len(predictions))

print(predictions)
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(scaled_train_X, train_y, train_size=0.75, random_state=42)
lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_validation, y_validation)



params = {'boosting_type': 'gbdt',

         'objective': 'regression',

         'metric': {'l2', 'l1'},

         'num_leaves': 31,

         'learning_rate': 0.05,

         'feature_fraction': 0.9,

         'bagging_fraction': 0.8,

         'bagging_freq': 5,

         'verbose': 0}

gbm = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=lgb_eval, early_stopping_rounds=5)
predictions = gbm.predict(scaled_test_X, num_iteration=gbm.best_iteration)

print(len(predictions))

print(predictions)
submission.time_to_failure = predictions

submission.to_csv('submission.csv', index=True)