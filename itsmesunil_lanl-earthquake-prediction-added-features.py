# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn import metrics

import lightgbm as lgb

import xgboost as xgb

from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats

import numpy as np

from scipy.stats import kurtosis

from scipy.stats import skew

from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold

# Any results you write to the current directory are saved as output.
%%time

train_df = pd.read_csv('../input/train.csv',dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
train_df.head()
print('The training data has {} rows and {} columns'.format(train_df.shape[0],train_df.shape[1]))
# there is a lot of data in the training set. Obviously need more computing power
#Skip 100 rows each

train_ad_sample_df = train_df['acoustic_data'].values[::100]

train_ttf_sample_df = train_df['time_to_failure'].values[::100]



def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='y')

    ax1.set_ylabel('acoustic data', color='y')

    plt.legend(['acoustic data'], bbox_to_anchor=(1.04,1.02), loc="upper left")

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='b')

    ax2.set_ylabel('quake time', color='b')

    plt.legend(['quake time'], bbox_to_anchor=(1.04,1.07), loc="upper left")

    plt.grid(True)



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
train_ad_sample_df = train_df['acoustic_data'].values[:6291455]

train_qt_sample_df = train_df['time_to_failure'].values[:6291455]

fig, ax = plt.subplots(2,1, figsize=(20,12))

ax[0].plot(train_qt_sample_df, c="y")

ax[0].set_title("Quaketime of 1% of total data")

ax[0].set_ylabel("Quaketime in ms");

ax[1].plot(train_ad_sample_df, c="mediumseagreen")

ax[1].set_title("Signal of 1% of total data")

ax[1].set_ylabel("Acoustic Signal");
fig, ax = plt.subplots(1,1,figsize=(10,5))

ax.plot(train_df.index.values[0:50000], train_df.time_to_failure.values[0:50000], c="Red")

ax.set_xlabel("Index")

ax.set_ylabel("Time to quake")

ax.set_title("How does the second quaketime pattern look like?")
#looking into the test data

from os import listdir

test_dir_path = '../input/test/'

test_files = listdir(test_dir_path)

print(len(test_files))
test_files[0:5]
seg = pd.read_csv('../input/test/seg_0b082e.csv')
seg.head()
len(seg)
#looking the submission file

sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission.head()
len(sample_submission.seg_id)
# EDA of test files

fig, ax = plt.subplots(4,1, figsize=(20,25))



for n in range(4):

    seg = pd.read_csv(test_dir_path  + test_files[n])

    ax[n].plot(seg.acoustic_data.values, c="mediumslateblue")

    ax[n].set_xlabel("Index")

    ax[n].set_ylabel("Signal")

    ax[n].set_ylim([-300, 300])

    ax[n].set_title("Test {}".format(test_files[n]));
rows = 150000

segments = int(np.floor(train_df.shape[0] / rows))

print("Number of segments: ", segments)
# list of features to be engineered

features = ['ave', 'std', 'max', 'min',

            'av_change_abs', 'av_change_rate', 'abs_max', 'abs_min',

            'std_first_50000', 'std_last_50000', 'std_first_10000', 'std_last_10000',

            'avg_first_50000', 'avg_last_50000', 'avg_first_10000', 'avg_last_10000',

            'min_first_50000', 'min_last_50000', 'min_first_10000', 'min_last_10000',

            'max_first_50000', 'max_last_50000', 'max_first_10000', 'max_last_10000']
def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]



def classic_sta_lta(x, length_sta, length_lta):

    sta = np.cumsum(x ** 2)

    # Convert to float

    sta = np.require(sta, dtype=np.float)

    # Copy for LTA

    lta = sta.copy()

    # Compute the STA and the LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta

    # Pad zeros

    sta[:length_lta - 1] = 0

    # Avoid division by zero by setting zero values to tiny float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny

    return sta / lta
X = pd.DataFrame(index=range(segments), dtype=np.float64)

Y = pd.DataFrame(index=range(segments), dtype=np.float64,

                       columns=['time_to_failure'])
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

    X.loc[seg_id, 'Imean'] = imagFFT.mean()

    X.loc[seg_id, 'Istd'] = imagFFT.std()

    X.loc[seg_id, 'Imax'] = imagFFT.max()

    X.loc[seg_id, 'Imin'] = imagFFT.min()

    X.loc[seg_id, 'Rmean_last_5000'] = realFFT[-5000:].mean()

    X.loc[seg_id, 'Rstd__last_5000'] = realFFT[-5000:].std()

    X.loc[seg_id, 'Rmax_last_5000'] = realFFT[-5000:].max()

    X.loc[seg_id, 'Rmin_last_5000'] = realFFT[-5000:].min()

    X.loc[seg_id, 'Rmean_last_15000'] = realFFT[-15000:].mean()

    X.loc[seg_id, 'Rstd_last_15000'] = realFFT[-15000:].std()

    X.loc[seg_id, 'Rmax_last_15000'] = realFFT[-15000:].max()

    X.loc[seg_id, 'Rmin_last_15000'] = realFFT[-15000:].min()

    

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

    

    X.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(xc), 0.95)

    X.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(xc), 0.99)

    X.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(xc), 0.05)

    X.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(xc), 0.01)

    

    X.loc[seg_id, 'trend'] = add_trend_feature(xc)

    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)

    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()

    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()

    

    X.loc[seg_id, 'mad'] = xc.mad()

    X.loc[seg_id, 'kurt'] = xc.kurtosis()

    X.loc[seg_id, 'skew'] = xc.skew()

    X.loc[seg_id, 'med'] = xc.median()

    

    X.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(xc)).mean()

    X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()

    X.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(xc, 500, 10000).mean()

    X.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(xc, 5000, 100000).mean()

    X.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(xc, 3333, 6666).mean()

    X.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(xc, 10000, 25000).mean()

    X.loc[seg_id, 'Moving_average_700_mean'] = xc.rolling(window=700).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_1500_mean'] = xc.rolling(window=1500).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_3000_mean'] = xc.rolling(window=3000).mean().mean(skipna=True)

    X.loc[seg_id, 'Moving_average_6000_mean'] = xc.rolling(window=6000).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    X.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(xc, span=300).mean()).mean(skipna=True)

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

from tqdm import tqdm_notebook

for segment in tqdm_notebook(range(segments)):

    seg = train_df.iloc[segment*rows:segment*rows+rows]

    create_features(segment, seg, X)

    Y.loc[segment, 'time_to_failure'] = seg['time_to_failure'].values[-1]
print(f'{X.shape[0]} samples in new train data and {X.shape[1]} columns.')
scaler = StandardScaler()

scaler.fit(X)

X_train_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')
X_test = pd.DataFrame(columns=X.columns, dtype=np.float64, index=submission.index)



for i, seg_id in enumerate(tqdm_notebook(X_test.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    create_features(seg_id, seg, X_test)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
def train_model(X=X_train_scaled, X_test=X_test_scaled, y=Y, params=None, folds=folds, model_type='lgb', 

                plot_feature_importance=False, model=None):



    oof = np.zeros(len(X))

    prediction = np.zeros(len(X_test))

    scores = []

    feature_importance = pd.DataFrame()

    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):

        print('Fold', fold_n, 'started at', time.ctime())

        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        

        if model_type == 'lgb':

            model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)

            model.fit(X_train, y_train, 

                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='mae',

                    verbose=10000, early_stopping_rounds=200)

            

            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

            

        if model_type == 'xgb':

            train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

            valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



            watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

            model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, 

                              verbose_eval=500, params=params)

            y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

            

        if model_type == 'rcv':

            model = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0), scoring='neg_mean_absolute_error', cv=3)

            model.fit(X_train, y_train)

            print(model.alpha_)



            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_absolute_error(y_valid, y_pred_valid)

            print(f'Fold {fold_n}. MAE: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'sklearn':

            model = model

            model.fit(X_train, y_train)

            

            y_pred_valid = model.predict(X_valid).reshape(-1,)

            score = mean_absolute_error(y_valid, y_pred_valid)

            print(f'Fold {fold_n}. MAE: {score:.4f}.')

            print('')

            

            y_pred = model.predict(X_test).reshape(-1,)

        

        if model_type == 'cat':

            model = CatBoostRegressor(iterations=20000,  eval_metric='MAE', **params)

            model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True, verbose=False)



            y_pred_valid = model.predict(X_valid)

            y_pred = model.predict(X_test)

        

        oof[valid_index] = y_pred_valid.reshape(-1,)

        scores.append(mean_absolute_error(y_valid, y_pred_valid))



        prediction += y_pred    

        

        if model_type == 'lgb':

            # feature importance

            fold_importance = pd.DataFrame()

            fold_importance["feature"] = X.columns

            fold_importance["importance"] = model.feature_importances_

            fold_importance["fold"] = fold_n + 1

            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)



    prediction /= n_fold

    

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    

    if model_type == 'lgb':

        feature_importance["importance"] /= n_fold

        if plot_feature_importance:

            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(

                by="importance", ascending=False)[:50].index



            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]



            plt.figure(figsize=(16, 12));

            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));

            plt.title('LGB Features (avg over folds)');

        

            return oof, prediction, feature_importance

        return oof, prediction

    

    else:

        return oof, prediction
params = {'num_leaves': 54,

         'min_data_in_leaf': 79,

         'objective': 'huber',

         'max_depth': -1,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         # "feature_fraction": 0.8354507676881442,

         "bagging_freq": 3,

         "bagging_fraction": 0.8126672064208567,

         "bagging_seed": 11,

         "metric": 'mae',

         "verbosity": -1,

         'reg_alpha': 1.1302650970728192,

         'reg_lambda': 0.3603427518866501

         }

oof_lgb, prediction_lgb, feature_importance = train_model(params=params, model_type='lgb', plot_feature_importance=True)
# xgb_params = {'eta': 0.03, 'max_depth': 10, 'subsample': 0.85, #'colsample_bytree': 0.8, 

#           'objective': 'reg:linear', 'eval_metric': 'mae', 'silent': True, 'nthread': 4}

# oof_xgb, prediction_xgb = train_model(params=xgb_params, model_type='xgb')
# from sklearn.svm import NuSVR, SVR

# model = NuSVR(gamma='scale', nu=0.75, C=10.0)

# oof_svr, prediction_svr = train_model(params=None, model_type='sklearn', model=model)
# prediction_lgb[:10], prediction_xgb[:10], prediction_svr[:10]
#submission['time_to_failure'] = (prediction_lgb + prediction_xgb + prediction_svr) / 3

submission['time_to_failure'] = prediction_lgb

print(submission.head())

submission.to_csv('submission.csv')