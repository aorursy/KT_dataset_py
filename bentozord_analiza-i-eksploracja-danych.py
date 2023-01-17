# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd 



import os



DATA_DIR = "../input"

TEST_DIR = r'../input/test'



train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), nrows=300000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})



print(train_df.info())





ld = os.listdir(TEST_DIR)

sizes = np.zeros(len(ld))



for i, f in enumerate(ld):

    df = pd.read_csv(os.path.join(TEST_DIR, f))

    sizes[i] = df.shape[0]



print(np.mean(sizes))

print(np.min(sizes))

print(np.max(sizes))

print('ok')
import seaborn as sns

import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.svm import SVR, NuSVR

from sklearn.kernel_ridge import KernelRidge

import pandas as pd

import numpy as np

import os

import gc

import warnings

warnings.filterwarnings("ignore")



DATA_DIR = "../input"

TEST_DIR = r'../input/test'



ld = os.listdir(TEST_DIR)

sizes = np.zeros(len(ld))



from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy.stats import pearsonr

from scipy import stats

from sklearn.kernel_ridge import KernelRidge



import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



from tsfresh.feature_extraction import feature_calculators



%matplotlib inline

sns.set_style('darkgrid')
acoustic_data_sample = train_df['acoustic_data'].values[::50]

time_to_failure_sample = train_df['time_to_failure'].values[::50]



fig, ax1 = plt.subplots(figsize = (12,8))

plt.title('Data from DF')

plt.plot(acoustic_data_sample, color='r')

ax1.set_ylabel('Acousting Data', color='r')

plt.legend(['acoustic data'], loc=(0.01, 0.95))

ax2 = ax1.twinx()

plt.plot(time_to_failure_sample, color='b')

ax2.set_ylabel('Time to Failure', color='b')

plt.legend(['time_to_failure'], loc=(0.01, 0.95))

plt.grid(True)



del acoustic_data_sample, time_to_failure_sample

gc.collect()
np.random.seed(2018)

rand_idxs = np.random.randint(0, 300000000-150000, size=9, dtype=np.int32)

fig, axes = plt.subplots(3, 3, figsize=(18, 10))



for x in range(3):

    for y in range(3):

        ad = train_df['acoustic_data'].values[rand_idxs[x*3 + y]: rand_idxs[x*3 + y] + 150000]

        ttf = train_df['time_to_failure'].values[rand_idxs[x*3 + y]: rand_idxs[x*3 + y] + 150000]



        axes[x][y].plot(ad, color='blue')

        axes[x][y].set_xticks([])



        s = axes[x][y].twinx()

        s.plot(ttf, color='red')

        

plt.tight_layout()

plt.show()

del ad, ttf, rand_idxs

gc.collect()
d = {'vals': train_df[train_df['acoustic_data']>2000].index.values}

peaks = pd.DataFrame(d)

peaks['diff'] = peaks['vals'].diff(periods=-1)

selected_peaks = peaks[abs(peaks['diff'])>30000]['vals'].values





train_df['diff'] = train_df['time_to_failure'].diff()

indexes_of_eartgquakes = train_df[train_df['diff']>1].index.values



print('Number of earthquakes in loaded data: ', len(indexes_of_eartgquakes))

print('Number of peaks in loaded data: ',len(selected_peaks))



print(selected_peaks)



del peaks, d

gc.collect()
fig, axes = plt.subplots(4, 2, figsize=(18, 10))

fig.delaxes(axes[3,1])



for x in range(7):

        ad = train_df['acoustic_data'].values[selected_peaks[x]-75000: selected_peaks[x]+75000]

        ttf = train_df['time_to_failure'].values[selected_peaks[x]-75000: selected_peaks[x]+75000]



        axes[int(x/2)][x%2].plot(ad, color='blue')

        axes[int(x/2)][x%2].set_xticks([])



        s = axes[int(x/2)][x%2].twinx()

        s.plot(ttf, color='red')

        

plt.tight_layout()

plt.show()

del ad, ttf

gc.collect()
fig, axes = plt.subplots(4, 2, figsize=(18, 10))

fig.delaxes(axes[3,1])



for x in range(7):

        ad = train_df['acoustic_data'].values[indexes_of_eartgquakes[x]-140000: indexes_of_eartgquakes[x]+10000]

        ttf = train_df['time_to_failure'].values[indexes_of_eartgquakes[x]-140000: indexes_of_eartgquakes[x]+10000]



        axes[int(x/2)][x%2].plot(ad, color='blue')

        axes[int(x/2)][x%2].set_xticks([])



        s = axes[int(x/2)][x%2].twinx()

        s.plot(ttf, color='red')

        

plt.tight_layout()

plt.show()

del ad, ttf

gc.collect()
def classic_sta_lta(x, length_sta, length_lta):

    

    sta = np.cumsum(x ** 2)



    # Zamiana na float

    sta = np.require(sta, dtype=np.float)



    # Kopia dla LTA

    lta = sta.copy()



    # Obliczanie STA i LTA

    sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]

    sta /= length_sta

    lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]

    lta /= length_lta



    # Uzupełnienie zerami

    sta[:length_lta - 1] = 0



    # Aby nie dzielić przez 0 ustawiamy 0 na małe liczby typu float

    dtiny = np.finfo(0.0).tiny

    idx = lta < dtiny

    lta[idx] = dtiny



    return sta / lta
def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)
percentiles = [1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]

hann_windows = [50, 150, 1500, 15000]

spans = [300, 3000, 30000, 50000]

windows = [10, 50, 100, 500, 1000, 10000]

borders = list(range(-4000, 4001, 1000))

peaks = [10, 20, 50, 100]

coefs = [1, 5, 10, 50, 100]

lags = [10, 100, 1000, 10000]

autocorr_lags = [5, 10, 50, 100, 500, 1000, 5000, 10000]
def gen_features(x, zero_mean=False):

    if zero_mean==True:

        x = x-x.mean()

    strain = {}

    strain['mean'] = x.mean()

    strain['std']=x.std()

    strain['max']=x.max()

    strain['kurtosis']=x.kurtosis()

    strain['skew']=x.skew()

    zc = np.fft.fft(x)

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)

    strain['min']=x.min()

    strain['sum']=x.sum()

    strain['mad']=x.mad()

    strain['median']=x.median()

    

    strain['mean_change_abs'] = np.mean(np.diff(x))

    strain['mean_change_rate'] = np.mean(np.nonzero((np.diff(x) / x[:-1]))[0])

    strain['abs_max'] = np.abs(x).max()

    strain['abs_min'] = np.abs(x).min()

    

    strain['avg_first_50000'] = x[:50000].mean()

    strain['avg_last_50000'] = x[-50000:].mean()

    strain['avg_first_10000'] = x[:10000].mean()

    strain['avg_last_10000'] = x[-10000:].mean()

    

    strain['min_first_50000'] = x[:50000].min()

    strain['min_last_50000'] = x[-50000:].min()

    strain['min_first_10000'] = x[:10000].min()

    strain['min_last_10000'] = x[-10000:].min()

    

    strain['max_first_50000'] = x[:50000].max()

    strain['max_last_50000'] = x[-50000:].max()

    strain['max_first_10000'] = x[:10000].max()

    strain['max_last_10000'] = x[-10000:].max()

    

    strain['max_to_min'] = x.max() / np.abs(x.min())

    strain['max_to_min_diff'] = x.max() - np.abs(x.min())

    strain['count_big'] = len(x[np.abs(x) > 500])

           

    strain['mean_change_rate_first_50000'] = calc_change_rate(x[:50000])

    strain['mean_change_rate_last_50000'] = calc_change_rate(x[-50000:])

    strain['mean_change_rate_first_10000'] = calc_change_rate(x[:10000])

    strain['mean_change_rate_last_10000'] = calc_change_rate(x[-10000:])

    

    strain['q95'] = np.quantile(x, 0.95)

    strain['q99'] = np.quantile(x, 0.99)

    strain['q05'] = np.quantile(x, 0.05)

    strain['q01'] = np.quantile(x, 0.01)

    

    strain['abs_q95'] = np.quantile(np.abs(x), 0.95)

    strain['abs_q99'] = np.quantile(np.abs(x), 0.99)

    strain['abs_q05'] = np.quantile(np.abs(x), 0.05)

    strain['abs_q01'] = np.quantile(np.abs(x), 0.01)

    

    for autocorr_lag in autocorr_lags:

        strain['autocorrelation_' + str(autocorr_lag)] = feature_calculators.autocorrelation(x, autocorr_lag)

    

    # percentiles on original and absolute values

    for p in percentiles:

        strain['percentile_'+str(p)] = np.percentile(x, p)

        strain['abs_percentile_'+str(p)] = np.percentile(np.abs(x), p)

    

#     strain['trend'] = add_trend_feature(x)

#     strain['abs_trend'] = add_trend_feature(x, abs_values=True)

    strain['abs_mean'] = np.abs(x).mean()

    strain['abs_std'] = np.abs(x).std()

    

    strain['quantile_0.95']=np.quantile(x, 0.95)

    strain['quantile_0.99']=np.quantile(x, 0.99)

    strain['quantile_0.05']=np.quantile(x, 0.05)

    strain['realFFT_mean']=realFFT.mean()

    strain['realFFT_std']=realFFT.std()

    strain['realFFT_max']=realFFT.max()

    strain['realFFT_min']=realFFT.min()

    strain['imagFFT_mean']=imagFFT.mean()

    strain['imagFFT_std']=realFFT.std()

    strain['imagFFT_max']=realFFT.max()

    strain['imaglFFT_min']=realFFT.min()

    

    strain['std_first_50000']=x[:50000].std()

    strain['std_last_50000']=x[-50000:].std()

    strain['std_first_25000']=x[:25000].std()

    strain['std_last_25000']=x[-25000:].std()

    strain['std_first_10000']=x[:10000].std()

    strain['std_last_10000']=x[-10000:].std()

    strain['std_first_5000']=x[:5000].std()

    strain['std_last_5000']=x[-5000:].std()

        

    strain['Hilbert_mean'] = np.abs(hilbert(x)).mean()

    strain['Hann_window_mean'] = (convolve(x, hann(150), mode='same') / sum(hann(150))).mean()

    strain['classic_sta_lta1_mean'] = classic_sta_lta(x, 500, 10000).mean()

    strain['classic_sta_lta2_mean'] = classic_sta_lta(x, 5000, 100000).mean()

    strain['classic_sta_lta3_mean'] = classic_sta_lta(x, 3333, 6666).mean()

    strain['classic_sta_lta4_mean'] = classic_sta_lta(x, 10000, 25000).mean()

    #strain['classic_sta_lta5_mean'] = classic_sta_lta(x, 50, 1000).mean() contains inf and Nan values

    strain['classic_sta_lta6_mean'] = classic_sta_lta(x, 100, 5000).mean()

    #strain['classic_sta_lta7_mean'] = classic_sta_lta(x, 333, 666).mean() contains inf and Nan values

    strain['classic_sta_lta8_mean'] = classic_sta_lta(x, 4000, 10000).mean()

    strain['Moving_average_700_mean'] = x.rolling(window=700).mean().mean(skipna=True)

    moving_average_700_mean = x.rolling(window=700).mean().mean(skipna=True)

    ewma = pd.Series.ewm

    strain['exp_Moving_average_300_mean'] = (ewma(x, span=300).mean()).mean(skipna=True)

    strain['exp_Moving_average_3000_mean'] = ewma(x, span=3000).mean().mean(skipna=True)

    strain['exp_Moving_average_30000_mean'] = ewma(x, span=30000).mean().mean(skipna=True)

    no_of_std = 3

    strain['MA_700MA_std_mean'] = x.rolling(window=700).std().mean()

    strain['MA_1000MA_std_mean'] = x.rolling(window=1000).std().mean()

    

    strain['iqr'] = np.subtract(*np.percentile(x, [75, 25]))

    strain['q999'] = np.quantile(x,0.999)

    strain['q001'] = np.quantile(x,0.001)

    strain['ave10'] = stats.trim_mean(x, 0.1)

        

    for window in windows:

        x_roll_std = x.rolling(window).std().dropna().values

        x_roll_mean = x.rolling(window).mean().dropna().values

        

        strain['ave_roll_std_' + str(window)] = x_roll_std.mean()

        strain['std_roll_std_' + str(window)] = x_roll_std.std()

        strain['max_roll_std_' + str(window)] = x_roll_std.max()

        strain['min_roll_std_' + str(window)] = x_roll_std.min()

        strain['q01_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.01)

        strain['q05_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.05)

        strain['q95_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.95)

        strain['q99_roll_std_' + str(window)] = np.quantile(x_roll_std, 0.99)

        strain['av_change_abs_roll_std_' + str(window)] = np.mean(np.diff(x_roll_std))

        strain['av_change_rate_roll_std_' + str(window)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

        strain['abs_max_roll_std_' + str(window)] = np.abs(x_roll_std).max()

        

        for p in percentiles:

            strain['percentile_roll_std_' + str(p) + '_window_' + str(window)] = np.percentile(x_roll_std, p)

            strain['percentile_roll_mean_' + str(p) + '_window_' + str(window)] = np.percentile(x_roll_mean, p)

        

        strain['ave_roll_mean_' + str(window)] = x_roll_mean.mean()

        strain['std_roll_mean_' + str(window)] = x_roll_mean.std()

        strain['max_roll_mean_' + str(window)] = x_roll_mean.max()

        strain['min_roll_mean_' + str(window)] = x_roll_mean.min()

        strain['q01_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.01)

        strain['q05_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.05)

        strain['q95_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.95)

        strain['q99_roll_mean_' + str(window)] = np.quantile(x_roll_mean, 0.99)

        strain['av_change_abs_roll_mean_' + str(window)] = np.mean(np.diff(x_roll_mean))

        strain['av_change_rate_roll_mean_' + str(window)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        strain['abs_max_roll_mean_' + str(window)] = np.abs(x_roll_mean).max()

        

        

    return pd.Series(strain)
del train_df

gc.collect()



train_df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'), iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})

X_train = pd.DataFrame()

X_train_zero_mean = pd.DataFrame()

y_train = pd.Series()



for df in train_df:

    features = gen_features(df['acoustic_data'])

    ch_zero_mean = gen_features(df['acoustic_data'], zero_mean=True)

    X_train = X_train.append(features, ignore_index=True)

    X_train_zero_mean = X_train_zero_mean.append(ch_zero_mean, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]), ignore_index=True)



X_train.head()
X_test = pd.DataFrame()

X_test_zero_mean = pd.DataFrame()



for i, f in enumerate(ld):

    df = pd.read_csv(os.path.join(TEST_DIR, f))

    features = gen_features(df['acoustic_data'])

    ch_zero_mean = gen_features(df['acoustic_data'], zero_mean=True)

    X_test = X_test.append(features, ignore_index=True)

    X_test_zero_mean = X_test_zero_mean.append(ch_zero_mean, ignore_index=True)
def plot_acc_agg_ttf_data(features, title="Averaged accoustic data and ttf"):

    fig, axes = plt.subplots(3,3, figsize=(30, 18))

    

    for i in range(9):

        plt.title('Averaged accoustic data ({}) and time to failure'.format(features[i]))

        axes[int(i/3)][i%3].plot(X_train[features[i]], color='r')

        axes[int(i/3)][i%3].set_xlabel('training samples')

        axes[int(i/3)][i%3].set_ylabel('acoustic data ({})'.format(features[i]), color='r')

        plt.legend(['acoustic data ({})'.format(features[i])], loc=(0.01, 0.95))

        ax2 = axes[int(i/3)][i%3].twinx()

        ax2.plot(y_train, color='b')

        ax2.set_ylabel('time to failure', color='b')

        plt.legend(['time to failure'], loc=(0.01, 0.9))

        plt.grid(True)
def plot_distplot_features(features, nlines=4, colors=['green', 'blue'], df1=X_train, df2=X_test):

    plt.figure()

    fig, ax = plt.subplots(nlines,2,figsize=(16,4*nlines))

    for i in range(len(features)):

        plt.subplot(nlines,2,i+1)

        plt.hist(df1[features[i]],color=colors[0],bins=50, label='train', alpha=0.5)

        plt.hist(df2[features[i]],color=colors[1],bins=50, label='test', alpha=0.5)

        plt.legend()

        plt.title(features[i])

    plt.show()
features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurtosis', 'skew']

plot_distplot_features(features)
features = ['mean', 'std', 'max', 'min', 'sum', 'mad', 'kurtosis', 'skew']

plot_distplot_features(features, df1=X_train_zero_mean, df2=X_test_zero_mean)
features = ['std_first_50000', 'std_last_50000', 'std_first_25000','std_last_25000', 'std_first_10000','std_last_10000']

plot_distplot_features(features,3)
features = ['std_first_50000', 'std_last_50000', 'std_first_25000','std_last_25000', 'std_first_10000','std_last_10000']

plot_distplot_features(features,3, df1=X_train_zero_mean, df2=X_test_zero_mean)
all_features = X_train.columns.values

np.random.seed(2019)

rand_feat_idx = np.random.randint(0, len(all_features), size=9, dtype=np.int32)

rand_labels = [all_features[x] for x in rand_feat_idx]



plot_acc_agg_ttf_data(rand_labels)
corelations = np.abs(X_train.corrwith(y_train)).sort_values(ascending=False)

corelations_df = pd.DataFrame(data=corelations, columns=['corr'])

print("Number of high corelated values: ",corelations_df[corelations_df['corr']>=0.3]['corr'].count())



high_corr = corelations_df[corelations_df['corr']>=0.3]

print(high_corr)

high_corr_labels = high_corr.reset_index()['index'].values

print(high_corr_labels)
plot_acc_agg_ttf_data(high_corr_labels[:9])