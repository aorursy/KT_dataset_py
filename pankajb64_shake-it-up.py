# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import ShuffleSplit

from sklearn.linear_model import HuberRegressor, LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



from tqdm import tqdm_notebook

from sklearn.preprocessing import StandardScaler

from sklearn.svm import NuSVR, SVR

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import ShuffleSplit



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy import stats



import matplotlib.pyplot as plt

import gc

import dask.dataframe as dd

import xgboost as xgb

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

#print("train shape", train.shape)

#pd.set_option("display.precision", 15)  # show more decimals

#train.head()
print("train shape", train.shape)
pd.set_option("display.precision", 15)  # show more decimals
#train.head(15)
def chunkize(n_chunks):

    rolling_mean = []

    last_time = []

    init_idx = 0

    for _ in range(n_chunks):  # 629M / 150k = 4194

        x = train.iloc[init_idx:init_idx + 150000]

        last_time.append(x.time_to_failure.values[-1])

        rolling_mean.append(x.acoustic_data.abs().mean())

        init_idx += 150000



    rolling_mean = np.array(rolling_mean)

    last_time = np.array(last_time)

    

    return rolling_mean, last_time
#rolling_mean, last_time = chunkize(4194)
#splitter = ShuffleSplit(n_splits=1, test_size=.2, random_state=42)
#train_index, test_index = list(splitter.split(rolling_mean, last_time))[0]
#lr = LinearRegression()
#lr.fit(rolling_mean.reshape(-1,1), last_time)
#test_files = list(os.listdir('../input/test'))
#test_files
#test_dfs = [pd.read_csv('../input/test/{}'.format(file)) for file in test_files]
#test_means = np.array([df.acoustic_data.mean() for df in test_dfs])
#pred_times = lr.predict(test_means.reshape(-1,1))
#output = pd.DataFrame()
#test_files[0][:-4]
#output['seg_id'] = [file[:-4] for file in test_files]
#output['time_to_failure'] = pred_times
#output.to_csv('output_lr.csv', index=False,header=True)
# Create a training file with simple derived features

rows = 150_000

segments = int(np.floor(629145480 / rows))



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



X_tr = pd.DataFrame(index=range(segments), dtype=np.float64)



y_tr = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['time_to_failure'])



total_mean = train['acoustic_data'].mean()

total_std = train['acoustic_data'].std()

total_max = train['acoustic_data'].max()

total_min = train['acoustic_data'].min()

total_sum = train['acoustic_data'].sum()

total_abs_sum = np.abs(train['acoustic_data']).sum()



for segment in tqdm_notebook(range(segments)):

    seg = train.iloc[segment*rows:segment*rows+rows]

    x = pd.Series(seg['acoustic_data'].values)

    y = seg['time_to_failure'].values[-1]

    

    y_tr.loc[segment, 'time_to_failure'] = y

    X_tr.loc[segment, 'mean'] = x.mean()

    X_tr.loc[segment, 'std'] = x.std()

    X_tr.loc[segment, 'max'] = x.max()

    X_tr.loc[segment, 'min'] = x.min()

    

    

    X_tr.loc[segment, 'mean_change_abs'] = np.mean(np.diff(x))

    X_tr.loc[segment, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    X_tr.loc[segment, 'abs_max'] = np.abs(x).max()

    X_tr.loc[segment, 'abs_min'] = np.abs(x).min()

    

    X_tr.loc[segment, 'std_first_50000'] = x[:50000].std()

    X_tr.loc[segment, 'std_last_50000'] = x[-50000:].std()

    X_tr.loc[segment, 'std_first_10000'] = x[:10000].std()

    X_tr.loc[segment, 'std_last_10000'] = x[-10000:].std()

    

    X_tr.loc[segment, 'avg_first_50000'] = x[:50000].mean()

    X_tr.loc[segment, 'avg_last_50000'] = x[-50000:].mean()

    X_tr.loc[segment, 'avg_first_10000'] = x[:10000].mean()

    X_tr.loc[segment, 'avg_last_10000'] = x[-10000:].mean()

    

    X_tr.loc[segment, 'min_first_50000'] = x[:50000].min()

    X_tr.loc[segment, 'min_last_50000'] = x[-50000:].min()

    X_tr.loc[segment, 'min_first_10000'] = x[:10000].min()

    X_tr.loc[segment, 'min_last_10000'] = x[-10000:].min()

    

    X_tr.loc[segment, 'max_first_50000'] = x[:50000].max()

    X_tr.loc[segment, 'max_last_50000'] = x[-50000:].max()

    X_tr.loc[segment, 'max_first_10000'] = x[:10000].max()

    X_tr.loc[segment, 'max_last_10000'] = x[-10000:].max()

    

    X_tr.loc[segment, 'max_to_min'] = x.max() / np.abs(x.min())

    X_tr.loc[segment, 'max_to_min_diff'] = x.max() - np.abs(x.min())

    X_tr.loc[segment, 'count_big'] = len(x[np.abs(x) > 500])

    X_tr.loc[segment, 'sum'] = x.sum()

    

    X_tr.loc[segment, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])

    X_tr.loc[segment, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])

    X_tr.loc[segment, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])

    X_tr.loc[segment, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])

    

    X_tr.loc[segment, 'q95'] = np.quantile(x, 0.95)

    X_tr.loc[segment, 'q99'] = np.quantile(x, 0.99)

    X_tr.loc[segment, 'q05'] = np.quantile(x, 0.05)

    X_tr.loc[segment, 'q01'] = np.quantile(x, 0.01)

    

    X_tr.loc[segment, 'abs_q95'] = np.quantile(np.abs(x), 0.95)

    X_tr.loc[segment, 'abs_q99'] = np.quantile(np.abs(x), 0.99)

    X_tr.loc[segment, 'abs_q05'] = np.quantile(np.abs(x), 0.05)

    X_tr.loc[segment, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

    

    X_tr.loc[segment, 'trend'] = add_trend_feature(x)

    X_tr.loc[segment, 'abs_trend'] = add_trend_feature(x, abs_values=True)

    X_tr.loc[segment, 'abs_mean'] = np.abs(x).mean()

    X_tr.loc[segment, 'abs_std'] = np.abs(x).std()

    

    X_tr.loc[segment, 'mad'] = x.mad()

    X_tr.loc[segment, 'kurt'] = x.kurtosis()

    X_tr.loc[segment, 'skew'] = x.skew()

    X_tr.loc[segment, 'med'] = x.median()

    

    X_tr.loc[segment, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()

    X_tr.loc[segment, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

    X_tr.loc[segment, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()

    X_tr.loc[segment, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()

    X_tr.loc[segment, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()

    X_tr.loc[segment, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()

    X_tr.loc[segment, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()

    X_tr.loc[segment, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()

    X_tr.loc[segment, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()

    X_tr.loc[segment, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

    X_tr.loc[segment, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    X_tr.loc[segment, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)

    X_tr.loc[segment, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)

    X_tr.loc[segment, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)

    no_of_std = 3

    X_tr.loc[segment, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()

    X_tr.loc[segment,'MA_700MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()

    X_tr.loc[segment,'MA_700MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_700MA_std_mean']).mean()

    X_tr.loc[segment, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()

    X_tr.loc[segment,'MA_400MA_BB_high_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] + no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()

    X_tr.loc[segment,'MA_400MA_BB_low_mean'] = (X_tr.loc[segment, 'Moving_average_700_mean'] - no_of_std * X_tr.loc[segment, 'MA_400MA_std_mean']).mean()

    X_tr.loc[segment, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()

    X_tr.drop('Moving_average_700_mean', axis=1, inplace=True)

    

    X_tr.loc[segment, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_tr.loc[segment, 'q999'] = np.quantile(x,0.999)

    X_tr.loc[segment, 'q001'] = np.quantile(x,0.001)

    X_tr.loc[segment, 'ave10'] = stats.trim_mean(x, 0.1)



    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        

        X_tr.loc[segment, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X_tr.loc[segment, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X_tr.loc[segment, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X_tr.loc[segment, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X_tr.loc[segment, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X_tr.loc[segment, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X_tr.loc[segment, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X_tr.loc[segment, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X_tr.loc[segment, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_tr.loc[segment, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X_tr.loc[segment, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        X_tr.loc[segment, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X_tr.loc[segment, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X_tr.loc[segment, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X_tr.loc[segment, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X_tr.loc[segment, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X_tr.loc[segment, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X_tr.loc[segment, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X_tr.loc[segment, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X_tr.loc[segment, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_tr.loc[segment, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X_tr.loc[segment, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
print(f'{X_tr.shape[0]} samples in new train data and {X_tr.shape[1]} columns.')
# fillna in new columns

classic_sta_lta5_mean_fill = X_tr.loc[X_tr['classic_sta_lta5_mean'] != -np.inf, 'classic_sta_lta5_mean'].mean()

X_tr.loc[X_tr['classic_sta_lta5_mean'] == -np.inf, 'classic_sta_lta5_mean'] = classic_sta_lta5_mean_fill

X_tr['classic_sta_lta5_mean'] = X_tr['classic_sta_lta5_mean'].fillna(classic_sta_lta5_mean_fill)

classic_sta_lta7_mean_fill = X_tr.loc[X_tr['classic_sta_lta7_mean'] != -np.inf, 'classic_sta_lta7_mean'].mean()

X_tr.loc[X_tr['classic_sta_lta7_mean'] == -np.inf, 'classic_sta_lta7_mean'] = classic_sta_lta7_mean_fill

X_tr['classic_sta_lta7_mean'] = X_tr['classic_sta_lta7_mean'].fillna(classic_sta_lta7_mean_fill)
scaler = StandardScaler()

scaler.fit(X_tr)

X_train_scaled = pd.DataFrame(scaler.transform(X_tr), columns=X_tr.columns)
del train

gc.collect()
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame(columns=X_tr.columns, dtype=np.float64, index=submission.index)

plt.figure(figsize=(22, 16))



for i, seg_id in enumerate(tqdm_notebook(X_test.index)):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    x = pd.Series(seg['acoustic_data'].values)

    X_test.loc[seg_id, 'mean'] = x.mean()

    X_test.loc[seg_id, 'std'] = x.std()

    X_test.loc[seg_id, 'max'] = x.max()

    X_test.loc[seg_id, 'min'] = x.min()

        

    X_test.loc[seg_id, 'mean_change_abs'] = np.mean(np.diff(x))

    X_test.loc[seg_id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    X_test.loc[seg_id, 'abs_max'] = np.abs(x).max()

    X_test.loc[seg_id, 'abs_min'] = np.abs(x).min()

    

    X_test.loc[seg_id, 'std_first_50000'] = x[:50000].std()

    X_test.loc[seg_id, 'std_last_50000'] = x[-50000:].std()

    X_test.loc[seg_id, 'std_first_10000'] = x[:10000].std()

    X_test.loc[seg_id, 'std_last_10000'] = x[-10000:].std()

    

    X_test.loc[seg_id, 'avg_first_50000'] = x[:50000].mean()

    X_test.loc[seg_id, 'avg_last_50000'] = x[-50000:].mean()

    X_test.loc[seg_id, 'avg_first_10000'] = x[:10000].mean()

    X_test.loc[seg_id, 'avg_last_10000'] = x[-10000:].mean()

    

    X_test.loc[seg_id, 'min_first_50000'] = x[:50000].min()

    X_test.loc[seg_id, 'min_last_50000'] = x[-50000:].min()

    X_test.loc[seg_id, 'min_first_10000'] = x[:10000].min()

    X_test.loc[seg_id, 'min_last_10000'] = x[-10000:].min()

    

    X_test.loc[seg_id, 'max_first_50000'] = x[:50000].max()

    X_test.loc[seg_id, 'max_last_50000'] = x[-50000:].max()

    X_test.loc[seg_id, 'max_first_10000'] = x[:10000].max()

    X_test.loc[seg_id, 'max_last_10000'] = x[-10000:].max()

    

    X_test.loc[seg_id, 'max_to_min'] = x.max() / np.abs(x.min())

    X_test.loc[seg_id, 'max_to_min_diff'] = x.max() - np.abs(x.min())

    X_test.loc[seg_id, 'count_big'] = len(x[np.abs(x) > 500])

    X_test.loc[seg_id, 'sum'] = x.sum()

    

    X_test.loc[seg_id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x[:50000]) / x[:50000][:-1]))[0])

    X_test.loc[seg_id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x[-50000:]) / x[-50000:][:-1]))[0])

    X_test.loc[seg_id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x[:10000]) / x[:10000][:-1]))[0])

    X_test.loc[seg_id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x[-10000:]) / x[-10000:][:-1]))[0])

    

    X_test.loc[seg_id, 'q95'] = np.quantile(x,0.95)

    X_test.loc[seg_id, 'q99'] = np.quantile(x,0.99)

    X_test.loc[seg_id, 'q05'] = np.quantile(x,0.05)

    X_test.loc[seg_id, 'q01'] = np.quantile(x,0.01)

    

    X_test.loc[seg_id, 'abs_q95'] = np.quantile(np.abs(x), 0.95)

    X_test.loc[seg_id, 'abs_q99'] = np.quantile(np.abs(x), 0.99)

    X_test.loc[seg_id, 'abs_q05'] = np.quantile(np.abs(x), 0.05)

    X_test.loc[seg_id, 'abs_q01'] = np.quantile(np.abs(x), 0.01)

    

    X_test.loc[seg_id, 'trend'] = add_trend_feature(x)

    X_test.loc[seg_id, 'abs_trend'] = add_trend_feature(x, abs_values=True)

    X_test.loc[seg_id, 'abs_mean'] = np.abs(x).mean()

    X_test.loc[seg_id, 'abs_std'] = np.abs(x).std()

    

    X_test.loc[seg_id, 'mad'] = x.mad()

    X_test.loc[seg_id, 'kurt'] = x.kurtosis()

    X_test.loc[seg_id, 'skew'] = x.skew()

    X_test.loc[seg_id, 'med'] = x.median()

    

    X_test.loc[seg_id, 'Hilbert_mean'] = np.abs(hilbert(x)).mean()

    X_test.loc[seg_id, 'Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

    X_test.loc[seg_id, 'classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()

    X_test.loc[seg_id, 'classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()

    X_test.loc[seg_id, 'classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()

    X_test.loc[seg_id, 'classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()

    X_test.loc[seg_id, 'classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean()

    X_test.loc[seg_id, 'classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()

    X_test.loc[seg_id, 'classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean()

    X_test.loc[seg_id, 'classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

    X_test.loc[seg_id, 'Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    X_test.loc[seg_id, 'exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)

    X_test.loc[seg_id, 'exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)

    X_test.loc[seg_id, 'exp_Moving_average_30000_mean'] = ewma(x, span=6000).mean().mean(skipna=True)

    no_of_std = 3

    X_test.loc[seg_id, 'MA_700MA_std_mean'] = x.rolling(window=700).std().mean()

    X_test.loc[seg_id,'MA_700MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X_test.loc[seg_id,'MA_700MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_700MA_std_mean']).mean()

    X_test.loc[seg_id, 'MA_400MA_std_mean'] = x.rolling(window=400).std().mean()

    X_test.loc[seg_id,'MA_400MA_BB_high_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] + no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X_test.loc[seg_id,'MA_400MA_BB_low_mean'] = (X_test.loc[seg_id, 'Moving_average_700_mean'] - no_of_std * X_test.loc[seg_id, 'MA_400MA_std_mean']).mean()

    X_test.loc[seg_id, 'MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()

    X_test.drop('Moving_average_700_mean', axis=1, inplace=True)

    

    X_test.loc[seg_id, 'iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    X_test.loc[seg_id, 'q999'] = np.quantile(x,0.999)

    X_test.loc[seg_id, 'q001'] = np.quantile(x,0.001)

    X_test.loc[seg_id, 'ave10'] = stats.trim_mean(x, 0.1)

    

    for windows in [10, 100, 1000]:

        x_roll_std = x.rolling(windows).std().dropna().values

        x_roll_mean = x.rolling(windows).mean().dropna().values

        

        X_test.loc[seg_id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()

        X_test.loc[seg_id, 'std_roll_std_' + str(windows)] = x_roll_std.std()

        X_test.loc[seg_id, 'max_roll_std_' + str(windows)] = x_roll_std.max()

        X_test.loc[seg_id, 'min_roll_std_' + str(windows)] = x_roll_std.min()

        X_test.loc[seg_id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)

        X_test.loc[seg_id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)

        X_test.loc[seg_id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)

        X_test.loc[seg_id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)

        X_test.loc[seg_id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))

        X_test.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        X_test.loc[seg_id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()

        

        X_test.loc[seg_id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()

        X_test.loc[seg_id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()

        X_test.loc[seg_id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()

        X_test.loc[seg_id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()

        X_test.loc[seg_id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)

        X_test.loc[seg_id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)

        X_test.loc[seg_id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)

        X_test.loc[seg_id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)

        X_test.loc[seg_id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))

        X_test.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X_test.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()

    

    if i < 12:

        plt.subplot(6, 4, i + 1)

        plt.plot(seg['acoustic_data'])

        plt.title(seg_id)



# fillna in new columns

X_test.loc[X_test['classic_sta_lta5_mean'] == -np.inf, 'classic_sta_lta5_mean'] = classic_sta_lta5_mean_fill

X_test.loc[X_test['classic_sta_lta7_mean'] == -np.inf, 'classic_sta_lta7_mean'] = classic_sta_lta7_mean_fill

X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
params = {'eta': 0.05,

              'max_depth': 10,

              'subsample': 0.9,

              'objective': 'reg:linear',

              'eval_metric': 'mae',

              'silent': True,

              'nthread': 4}
X=X_train_scaled

X_test=X_test_scaled
splitter = ShuffleSplit(n_splits=1, test_size=.2)
train_index, valid_index = list(splitter.split(X))[0]
y = y_tr
X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]

y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)

valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)



watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]

model = xgb.train(dtrain=train_data, num_boost_round=20000, evals=watchlist, early_stopping_rounds=200, verbose_eval=500, params=params)

y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)
mean_absolute_error(y_valid, y_pred_valid)
submission['time_to_failure'] = y_pred

# submission['time_to_failure'] = prediction_lgb_stack

print(submission.head())

submission.to_csv('submission_feats_xgb.csv')