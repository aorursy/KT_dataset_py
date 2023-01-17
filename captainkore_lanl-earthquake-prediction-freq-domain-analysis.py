



#Load data analysis/plotting modules

from tqdm import tqdm_notebook as tqdm

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt

import matplotlib as mpl

import lightgbm as lgb

import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib

import scipy

import numpy

import os









# only read first 300 million rows to conserve memory.

df_train = pd.read_csv("../input/train.csv", nrows=150e6)



test_files = os.listdir("../input/test/")

print(test_files[0:5])

# load sample_submission.csv

df_samplesub = pd.read_csv("../input/sample_submission.csv", index_col='seg_id')

pd.options.display.precision = 15



df_train.rename({'acoustic_data': 'ad', 'time_to_failure': 'ttf'}, axis=1, inplace=True)

df_train.head()
# from "LANL Earthquake EDA and Prediction" by Gabriel Preda



df_train_ad_1per_all = df_train['ad'].values[::100]

df_train_ttf_1per_all = df_train['ttf'].values[::100]



fig, ax1 = plt.subplots(figsize=(12, 8))

plt.title("Acoustic Data and Time to Failure")

plt.plot(df_train_ad_1per_all, color='b')

ax1.set_ylabel('acoustic data', color='b')

plt.legend(['acoustic data'], loc=(0.01, 0.95))

ax2 = ax1.twinx()

plt.plot(df_train_ttf_1per_all, color='r')

ax2.set_ylabel('time to failure', color='r')

plt.legend(['time to failure'], loc=(0.01, 0.9))

plt.grid(True)
df_train_sample = df_train[0:150000].ad

df_train_sample.head()
#get the power spectrum for the sample signal

import scipy.signal

fs = 4e6

def DFT(sig_in):



    fs = 4e6 #4e6 is the signal sample rate

    ps = np.abs(np.fft.fft(sig_in))**2

    freqs = np.fft.fftfreq(sig_in.size, 1/fs)

    return freqs, ps



freqs, ps = DFT(df_train_sample)

freqs_pdg, ps_pdg = scipy.signal.periodogram(df_train_sample.T, fs, scaling = 'spectrum', window = 'triang')

freqs_welch, ps_welch = scipy.signal.welch(df_train_sample.T, fs, scaling = 'spectrum')
# DFT

fig = plt.figure(figsize=(30, 20))

ax = fig.add_subplot(3, 1, 1)

ax.margins(x=0.003)

plt.plot(freqs,ps)

plt.xlim(left=0)

plt.ylim(0,5e8)

plt.title('Power Spectrum (DFT)', fontsize=24, loc='center')

plt.xlabel('Frequency (Hz)', fontsize = 18)

plt.ylabel('Power (signal strength)', fontsize = 18)



# Spectrogram 

#fig = plt.figure(figsize=(30, 20))

ax = fig.add_subplot(3, 1, 2)

ax.margins(x=0.003)

plt.plot(freqs_pdg, ps_pdg)

plt.title('Power Spectrum (Periodogram)', fontsize=24, loc='center')

plt.xlabel('Frequency (Hz)', fontsize = 18)

plt.ylabel('Power (signal strength)', fontsize = 18)



# Welch method

#fig = plt.figure(figsize=(30, 20))

ax = fig.add_subplot(3, 1, 3)

ax.margins(x=0.003)

plt.plot(freqs_welch, ps_welch)

plt.title('Power Spectrum (Welch method)', fontsize=24, loc='center')

plt.xlabel('Frequency (Hz)', fontsize = 18)

plt.ylabel('Power (signal strength)', fontsize = 18)

!pip install PeakUtils
import peakutils



i = 29

df_train_sampled = df_train[i*150000:(i*150000)+150000].ad



freqs, ps = DFT(df_train_sampled)

peaks_ind1 = peakutils.indexes(ps, thres=0.0001, min_dist=200 )

plt.margins(x=0.003)

plt.plot(freqs, ps)

for p in peaks_ind1:

    plt.scatter(freqs[p], ps[p], marker='s', color='red', label='v1')

plt.xlim(left=0)

plt.ylim([0,0.4e12])

plt.show()


def plot_ps():

    sns.set_style('whitegrid')

    plt.figure()

    fig, ax = plt.subplots(3, 4, figsize=(15, 15))

    fig.subplots_adjust(hspace=0.4, wspace=0.3)

    

    for i in tqdm(range(12)):

        df_train_sampled = df_train[i*150000:(i*150000)+150000].ad

        freqs, ps = DFT(df_train_sampled)

        peaks_ind = peakutils.indexes(ps, thres=1e-4, min_dist=200)

        plt.subplot(3, 4, i+1)

        plt.margins(x=0.003)

        plt.plot(freqs, ps)

        for p in peaks_ind:

            plt.scatter(freqs[p], ps[p], marker='s', color='red', label='v1')

        plt.xlim(0,500000)

        plt.ylim(0,1e9)

    plt.show()

    

plot_ps()

from scipy.signal import spectrogram



def make_spectrogram(sig_in):

    nperseg = 512 # default 256

    noverlap = nperseg // 4 # default: nperseg // 8

    fs = 4e6  # raw signal sample rate is 4MHz

    window = 'boxcar'

    scaling = 'spectrum' # {'density', 'spectrum'}

    detrend = 'linear' # {'linear', 'constant', False}

    f, t, Sxx = spectrogram(sig_in.T, nperseg=nperseg, noverlap=noverlap,

                                   fs=fs, window=window,

                                   scaling=scaling, detrend=detrend)

    return f, t, Sxx





f, t, Sxx_out = make_spectrogram(df_train_sample)



print('Sxx_out:', Sxx_out.shape)

print('f:', f.shape)

print('t:', t.shape)

fig = plt.figure(figsize=(30, 20))

ax = fig.add_subplot(4, 1, 1)

ax.margins(x=0.003)

plt.plot(df_train_sample)

plt.title('sample signal:', fontsize=18, loc='left')





ax = fig.add_subplot(4, 1, 2)

cmap = plt.get_cmap('magma')

spec = plt.pcolormesh(t, f, Sxx_out, cmap=cmap, norm = matplotlib.colors.Normalize(0,1))

plt.title('normalized log spectrogram:',

          fontsize=18, loc='left')

ax.set_ylim([0,5e5])



signal = Sxx_out[0:257,10]

peaks_ind = peakutils.indexes(signal, thres=0.05, min_dist=1)

plt.plot(f[0:257], signal)

for p in peaks_ind:

    plt.scatter(f[p], signal[p], marker='s', color='red', label='v1')
  

# getting errors during feature engineering where peaks couldn't be found for certain segments, 

# so we'll set the threshold really low to avoid that possibility.



def peak_finder(freqs, ps ,thres=0.00001, min_dist=200):

    cb = np.array([d[0] for d in ps])

    peaks_ind = peakutils.indexes(cb, thres=thres, min_dist=min_dist)

    peaks_data = pd.DataFrame(index=range(peaks_ind.size), dtype=np.float64, columns=['peak_freq','peak_power'])

    for p in range(peaks_ind.size):

            peaks_data['peak_freq'].loc[p] = freqs[peaks_ind[p]]

            peaks_data['peak_power'].loc[p] = ps[peaks_ind[p]][0]

    peaks_data.replace(["NaN", 'NaT'], np.nan, inplace = True)

    peaks_data = peaks_data.dropna()

    return peaks_data





def specgram_peak_slice(Sxx_out, f, t, thres=0.00001, min_dist=1):

    signal = Sxx_out[0][0:257,t]

    peaks_ind = peakutils.indexes(signal, thres=thres, min_dist=min_dist)

    peaks_data = pd.DataFrame(index=range(peaks_ind.size), dtype=np.float64, columns=['peak_freq','peak_power'])

    for p in range(peaks_ind.size):

        if f[peaks_ind[p]] > 0:

            peaks_data['peak_freq'].loc[p] = f[peaks_ind[p]]

            peaks_data['peak_power'].loc[p] = Sxx_out[0][peaks_ind[p]][t]

            

    peaks_data.replace(["NaN", 'NaT'], np.nan, inplace = True)

    peaks_data = peaks_data.dropna()

    return peaks_data



def specgram_temporal_features(power, f):

    tmaxfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tminfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmeanfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tstdfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmaxpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tminpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmeanpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tstdpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmadfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tkurtfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tskewfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmedfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmadpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tkurtpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tskewpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmedpow = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tmaxpowfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    tminpowfreq = pd.DataFrame(index=range(power.shape[1]), dtype=np.float64, columns=['val'])

    

    for seg in range(power.shape[1]):

        peak_slice = specgram_peak_slice(power, f, seg)

        

        tmaxfreq['val'].loc[seg] = peak_slice['peak_freq'].max()

        tminfreq['val'].loc[seg] = peak_slice['peak_freq'].min()

        tmeanfreq['val'].loc[seg] = peak_slice['peak_freq'].mean()

        tstdfreq['val'].loc[seg] = peak_slice['peak_freq'].std()



        tmaxpow['val'].loc[seg] = peak_slice['peak_power'].max()

        tminpow['val'].loc[seg] = peak_slice['peak_power'].min()

        tmeanpow['val'].loc[seg] = peak_slice['peak_power'].mean()

        tstdpow['val'].loc[seg] = peak_slice['peak_power'].std()



        tmadfreq['val'].loc[seg] = peak_slice['peak_freq'].mad()

        tkurtfreq['val'].loc[seg] = peak_slice['peak_freq'].kurtosis()

        tskewfreq['val'].loc[seg] = peak_slice['peak_freq'].skew()

        tmedfreq['val'].loc[seg] = peak_slice['peak_freq'].median()



        tmadpow['val'].loc[seg] = peak_slice['peak_power'].mad()

        tkurtpow['val'].loc[seg] = peak_slice['peak_power'].kurtosis()

        tskewpow['val'].loc[seg] = peak_slice['peak_power'].skew()

        tmedpow['val'].loc[seg] = peak_slice['peak_power'].median()



        tmaxpowfreq['val'].loc[seg] = peak_slice['peak_freq'][peak_slice['peak_power'].idxmax()]

        tminpowfreq['val'].loc[seg] = peak_slice['peak_freq'][peak_slice['peak_power'].idxmin()]



    return  tmaxfreq, tminfreq, tmeanfreq, tstdfreq, tmaxpow, tminpow, tmeanpow, tstdpow, tmadfreq, tkurtfreq, tskewfreq, tmedfreq, tmadpow, tkurtpow, tskewpow, tmedpow, tmaxpowfreq, tminpowfreq

        

def add_trend_feature(arr, abs_values=False):

    idx = np.array(range(len(arr)))

    if abs_values:

        arr = np.abs(arr)

    lr = LinearRegression()

    lr.fit(idx.reshape(-1, 1), arr)

    return lr.coef_[0]
segments = 500  #could only load 1000 segments worth of training data due to RAM restrictions.

X_train = pd.DataFrame(index=range(segments), dtype=np.float64)

y_train = pd.DataFrame(index=range(segments), dtype=np.float64, columns=['ttf'])
def feature_generation(seg_id, seg, X):

        

    xc = pd.DataFrame(seg['ad'])

    freqs, ps = DFT(xc)

    avg_peaks = peak_finder(freqs, ps) #sorted at low to high frequency

    f, t, power = make_spectrogram(xc)

    

    

#     average FFT peak features

    

    X.loc[seg_id, 'aFFT_max'] = avg_peaks['peak_freq'].max()

    X.loc[seg_id, 'aFFT_min'] = avg_peaks['peak_freq'].min()

    X.loc[seg_id, 'aFFT_mean'] = avg_peaks['peak_freq'].mean()

    X.loc[seg_id, 'aFFT_std'] = avg_peaks['peak_freq'].std()

    

    

    X.loc[seg_id, 'aFFT_max_pow'] = avg_peaks['peak_power'].max()

    X.loc[seg_id, 'aFFT_min_pow'] = avg_peaks['peak_power'].min()

    X.loc[seg_id, 'aFFT_mean_pow'] = avg_peaks['peak_power'].mean()

    X.loc[seg_id, 'aFFT_std_pow'] = avg_peaks['peak_power'].std()

    

    X.loc[seg_id, 'aFFT_mad_freq'] = avg_peaks['peak_freq'].mad()

    X.loc[seg_id, 'aFFT_kurt_freq'] = avg_peaks['peak_freq'].kurtosis()

    X.loc[seg_id, 'aFFT_skew_freq'] = avg_peaks['peak_freq'].skew()

    X.loc[seg_id, 'aFFT_med_freq'] = avg_peaks['peak_freq'].median()

    

    X.loc[seg_id, 'aFFT_mad_pow'] = avg_peaks['peak_power'].mad()

    X.loc[seg_id, 'aFFT_kurt_pow'] = avg_peaks['peak_power'].kurtosis()

    X.loc[seg_id, 'aFFT_skew_pow'] = avg_peaks['peak_power'].skew()

    X.loc[seg_id, 'aFFT_med_pow'] = avg_peaks['peak_power'].median()

    

    

    X.loc[seg_id, 'aFFT_max_pow_freq'] = avg_peaks['peak_freq'][avg_peaks['peak_power'].idxmax()]

    X.loc[seg_id, 'aFFT_min_pow_freq'] = avg_peaks['peak_freq'][avg_peaks['peak_power'].idxmin()]

    

    X.loc[seg_id, 'aFFT_mean_pow_q99'] = np.quantile(avg_peaks['peak_power'], 0.99)

    X.loc[seg_id, 'aFFT_mean_pow_q95'] = np.quantile(avg_peaks['peak_power'], 0.95)

    X.loc[seg_id, 'aFFT_mean_pow_q05'] = np.quantile(avg_peaks['peak_power'], 0.05)

    X.loc[seg_id, 'aFFT_mean_pow_q01'] = np.quantile(avg_peaks['peak_power'], 0.01)

    

    X.loc[seg_id, 'aFFT_mean_freq_q99'] = np.quantile(avg_peaks['peak_freq'], 0.99)

    X.loc[seg_id, 'aFFT_mean_freq_q95'] = np.quantile(avg_peaks['peak_freq'], 0.95)

    X.loc[seg_id, 'aFFT_mean_freq_q05'] = np.quantile(avg_peaks['peak_freq'], 0.05)

    X.loc[seg_id, 'aFFT_mean_freq_q01'] = np.quantile(avg_peaks['peak_freq'], 0.01)

    

#     Spectrogram FFT features (focusing on temporal characteristics and trend features)



    

    tmaxfreq, tminfreq, tmeanfreq, tstdfreq, tmaxpow, tminpow, tmeanpow, tstdpow, tmadfreq, tkurtfreq, tskewfreq, tmedfreq, tmadpow, tkurtpow, tskewpow, tmedpow, tmaxpowfreq, tminpowfreq  = specgram_temporal_features(power, f)

    

    X.loc[seg_id, 'tmaxfreq_trend'] = add_trend_feature(tmaxfreq['val'])

    X.loc[seg_id, 'tminfreq_trend'] = add_trend_feature(tminfreq['val'])

    X.loc[seg_id, 'tmeanfreq_trend'] = add_trend_feature(tmeanfreq['val'])

    X.loc[seg_id, 'tstdfreq_trend'] = add_trend_feature(tstdfreq['val'])

    X.loc[seg_id, 'tmaxpow_trend'] = add_trend_feature(tmaxpow['val'])

    X.loc[seg_id, 'tminpow_trend'] = add_trend_feature(tminpow['val'])

    X.loc[seg_id, 'tmeanpow_trend'] = add_trend_feature(tmeanpow['val'])

    X.loc[seg_id, 'tstdpow_trend'] = add_trend_feature(tstdpow['val'])

    X.loc[seg_id, 'tmadfreq_trend'] = add_trend_feature(tmadfreq['val'])

    X.loc[seg_id, 'tkurtfreq_trend'] = add_trend_feature(tkurtfreq['val'])

    X.loc[seg_id, 'tskewfreq_trend'] = add_trend_feature(tskewfreq['val'])

    X.loc[seg_id, 'tmedfreq_trend'] = add_trend_feature(tmedfreq['val'])

    X.loc[seg_id, 'tmadpow_trend'] = add_trend_feature(tmadpow['val'])

    X.loc[seg_id, 'tkurtpow_trend'] = add_trend_feature(tkurtpow['val'])

    X.loc[seg_id, 'tskewpow_trend'] = add_trend_feature(tskewpow['val'])

    X.loc[seg_id, 'tmedpow_trend'] = add_trend_feature(tmedpow['val'])

    X.loc[seg_id, 'tmaxpowfreq_trend'] = add_trend_feature(tmaxpowfreq['val'])

    X.loc[seg_id, 'tmaxpowfreq_trend'] = add_trend_feature(tminpowfreq['val'])

    

    X.loc[seg_id, 'tmaxfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmaxfreq['val'])))

    X.loc[seg_id, 'tminfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tminfreq['val'])))

    X.loc[seg_id, 'tmeanfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmeanfreq['val'])))

    X.loc[seg_id, 'tstdfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tstdfreq['val'])))

    X.loc[seg_id, 'tmaxpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmaxpow['val'])))

    X.loc[seg_id, 'tminpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tminpow['val'])))

    X.loc[seg_id, 'tmeanpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmeanpow['val'])))

    X.loc[seg_id, 'tstdpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tstdpow['val'])))

    X.loc[seg_id, 'tmadfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmadfreq['val'])))

    X.loc[seg_id, 'tkurtfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tkurtfreq['val'])))

    X.loc[seg_id, 'tskewfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tskewfreq['val'])))

    X.loc[seg_id, 'tmedfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmedfreq['val'])))

    X.loc[seg_id, 'tmadpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmadpow['val'])))

    X.loc[seg_id, 'tkurtpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tkurtpow['val'])))

    X.loc[seg_id, 'tskewpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tskewpow['val'])))

    X.loc[seg_id, 'tmedpow_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmedpow['val'])))

    X.loc[seg_id, 'tmaxpowfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tmaxpowfreq['val'])))

    X.loc[seg_id, 'tmaxpowfreq_grad_skew'] = scipy.stats.skew(np.diff(np.diff(tminpowfreq['val'])))

    

    X.loc[seg_id, 'tmaxfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmaxfreq['val'])))

    X.loc[seg_id, 'tminfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tminfreq['val'])))

    X.loc[seg_id, 'tmeanfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmeanfreq['val'])))

    X.loc[seg_id, 'tstdfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tstdfreq['val'])))

    X.loc[seg_id, 'tmaxpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmaxpow['val'])))

    X.loc[seg_id, 'tminpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tminpow['val'])))

    X.loc[seg_id, 'tmeanpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmeanpow['val'])))

    X.loc[seg_id, 'tstdpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tstdpow['val'])))

    X.loc[seg_id, 'tmadfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmadfreq['val'])))

    X.loc[seg_id, 'tkurtfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tkurtfreq['val'])))

    X.loc[seg_id, 'tskewfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tskewfreq['val'])))

    X.loc[seg_id, 'tmedfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmedfreq['val'])))

    X.loc[seg_id, 'tmadpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmadpow['val'])))

    X.loc[seg_id, 'tkurtpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tkurtpow['val'])))

    X.loc[seg_id, 'tskewpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tskewpow['val'])))

    X.loc[seg_id, 'tmedpow_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmedpow['val'])))

    X.loc[seg_id, 'tmaxpowfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tmaxpowfreq['val'])))

    X.loc[seg_id, 'tmaxpowfreq_grad_kurt'] = scipy.stats.kurtosis(np.diff(np.diff(tminpowfreq['val'])))

    

# iterate over 100 segments



rows = 150000

for seg_id in tqdm(range(100)):

    seg = df_train.iloc[seg_id*rows:seg_id*rows+rows]

    feature_generation(seg_id, seg, X_train)

    y_train.loc[seg_id, 'ttf'] = seg['ttf'].values[-1]



X_train.shape
X_train.head()
scaler = StandardScaler()

scaler.fit(X_train)

scaled_X_train = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
scaled_X_train.head()
X_test = pd.DataFrame(columns=X_train.columns, dtype=np.float64, index=df_samplesub.index)
test_PATH = "../input/test/"

for seg_id in tqdm(X_test.index):

    

    seg = pd.read_csv(test_PATH + seg_id + '.csv')

    seg.rename({'acoustic_data': 'ad'}, axis=1, inplace=True)

    feature_generation(seg_id, seg, X_test)

    

X_test.head()
scaled_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

scaled_X_test.head()
from sklearn.model_selection import KFold

n_fold = 5

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)

train_columns = scaled_X_train.columns.values
params = {'num_leaves': 51,

         'min_data_in_leaf': 10, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.001,

         "boosting": "gbdt",

         "feature_fraction": 0.91,

         "bagging_freq": 1,

         "bagging_fraction": 0.91,

         "bagging_seed": 42,

         "metric": 'mae',

         "lambda_l1": 0.1,

         "verbosity": -1,

         "nthread": -1,

         "random_state": 42}
oof = np.zeros(len(scaled_X_train))

predictions = np.zeros(len(scaled_X_test))

feature_importance_df = pd.DataFrame()

#run model

for fold_, (trn_idx, val_idx) in enumerate(folds.split(scaled_X_train,y_train.values)):

    strLog = "fold {}".format(fold_)

    print(strLog)

    

    X_tr, X_val = scaled_X_train.iloc[trn_idx], scaled_X_train.iloc[val_idx]

    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]



    model = lgb.LGBMRegressor(**params, n_estimators = 20000, n_jobs = -1)

    model.fit(X_tr, 

              y_tr, 

              eval_set=[(X_tr, y_tr), (X_val, y_val)], 

              eval_metric='mae',

              verbose=1000, 

              early_stopping_rounds=500)

    oof[val_idx] = model.predict(X_val, num_iteration=model.best_iteration_)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += model.predict(scaled_X_test, num_iteration=model.best_iteration_) / folds.n_splits
cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:200].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
df_samplesub.time_to_failure = predictions

df_samplesub.to_csv('submission.csv',index=True)