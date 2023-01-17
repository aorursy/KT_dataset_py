import os

import time

import warnings

import traceback

import numpy as np

import pandas as pd

from scipy import stats

import scipy.signal as sg

import multiprocessing as mp

from scipy.signal import hann

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler



import xgboost as xgb

import lightgbm as lgb

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error



from tqdm import tqdm_notebook

warnings.filterwarnings("ignore")
print(os.listdir("../input"))

print(os.listdir("../input/lanl-vettejeep-createfeature-1"))

print(os.listdir("../input/lanl-vettejeep-createfeature-2"))
OUTPUT_DIR = ''

DATA_DIR = '../input/LANL-Earthquake-Prediction'



SIG_LEN = 150000

NUM_SEG_PER_PROC = 4000

NUM_THREADS = 6



NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.

CUTOFF = 18000

MAX_FREQ_IDX = 20000

FREQ_STEP = 2500
datadir = '../input/lanl-vettejeep-createfeature-1'

df0 = pd.read_csv(os.path.join(datadir, 'train_x_%d.csv' % 0))

df1 = pd.read_csv(os.path.join(datadir, 'train_x_%d.csv' % 1))

df2 = pd.read_csv(os.path.join(datadir, 'train_x_%d.csv' % 2))

datadir = '../input/lanl-vettejeep-createfeature-2'

df3 = pd.read_csv(os.path.join(datadir, 'train_x_%d.csv' % 3))

df4 = pd.read_csv(os.path.join(datadir, 'train_x_%d.csv' % 4))

df5 = pd.read_csv(os.path.join(datadir, 'train_x_%d.csv' % 5))
df = pd.concat([df0,df1,df2,df3,df4,df5], axis=0, ignore_index=True)

print(df.shape)

display(df.head(5))

display(df.tail(5))
df.to_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'), index=False)
datadir = '../input/lanl-vettejeep-createfeature-1'

df0 = pd.read_csv(os.path.join(datadir, 'train_y_%d.csv' % 0))

df1 = pd.read_csv(os.path.join(datadir, 'train_y_%d.csv' % 1))

df2 = pd.read_csv(os.path.join(datadir, 'train_y_%d.csv' % 2))

datadir = '../input/lanl-vettejeep-createfeature-2'

df3 = pd.read_csv(os.path.join(datadir, 'train_y_%d.csv' % 3))

df4 = pd.read_csv(os.path.join(datadir, 'train_y_%d.csv' % 4))

df5 = pd.read_csv(os.path.join(datadir, 'train_y_%d.csv' % 5))
df_y = pd.concat([df0,df1,df2,df3,df4,df5], axis=0, ignore_index=True)

print(df_y.shape)

display(df_y.tail(5))
df_y.to_csv(os.path.join(OUTPUT_DIR, 'train_y.csv'), index=False)
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
def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter

    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)

    return b, a



def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter

    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')

    return b, a



def des_bw_filter_bp(low, high):  # band pass filter

    b, a = sg.butter(4, Wn=(low/NY_FREQ_IDX, high/NY_FREQ_IDX), btype='bandpass')

    return b, a
# Feature Creation Function

def create_features(seg_id, seg, X, st, end):

    try:

        X.loc[seg_id, 'seg_id'] = np.int32(seg_id)

        X.loc[seg_id, 'seg_start'] = np.int32(st)

        X.loc[seg_id, 'seg_end'] = np.int32(end)

    except:

        pass



    xc = pd.Series(seg['acoustic_data'].values)

    xcdm = xc - np.mean(xc)



    b, a = des_bw_filter_lp(cutoff=18000)

    xcz = sg.lfilter(b, a, xcdm)



    zc = np.fft.fft(xcz)

    zc = zc[:MAX_FREQ_IDX]



    # FFT transform values

    realFFT = np.real(zc)

    imagFFT = np.imag(zc)



    freq_bands = [x for x in range(0, MAX_FREQ_IDX, FREQ_STEP)]

    magFFT = np.sqrt(realFFT ** 2 + imagFFT ** 2)

    phzFFT = np.arctan(imagFFT / realFFT)

    phzFFT[phzFFT == -np.inf] = -np.pi / 2.0

    phzFFT[phzFFT == np.inf] = np.pi / 2.0

    phzFFT = np.nan_to_num(phzFFT)



    for freq in freq_bands:

        X.loc[seg_id, 'FFT_Mag_01q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.01)

        X.loc[seg_id, 'FFT_Mag_10q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.1)

        X.loc[seg_id, 'FFT_Mag_90q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.9)

        X.loc[seg_id, 'FFT_Mag_99q%d' % freq] = np.quantile(magFFT[freq: freq + FREQ_STEP], 0.99)

        X.loc[seg_id, 'FFT_Mag_mean%d' % freq] = np.mean(magFFT[freq: freq + FREQ_STEP])

        X.loc[seg_id, 'FFT_Mag_std%d' % freq] = np.std(magFFT[freq: freq + FREQ_STEP])

        X.loc[seg_id, 'FFT_Mag_max%d' % freq] = np.max(magFFT[freq: freq + FREQ_STEP])



        X.loc[seg_id, 'FFT_Phz_mean%d' % freq] = np.mean(phzFFT[freq: freq + FREQ_STEP])

        X.loc[seg_id, 'FFT_Phz_std%d' % freq] = np.std(phzFFT[freq: freq + FREQ_STEP])



    X.loc[seg_id, 'FFT_Rmean'] = realFFT.mean()

    X.loc[seg_id, 'FFT_Rstd'] = realFFT.std()

    X.loc[seg_id, 'FFT_Rmax'] = realFFT.max()

    X.loc[seg_id, 'FFT_Rmin'] = realFFT.min()

    X.loc[seg_id, 'FFT_Imean'] = imagFFT.mean()

    X.loc[seg_id, 'FFT_Istd'] = imagFFT.std()

    X.loc[seg_id, 'FFT_Imax'] = imagFFT.max()

    X.loc[seg_id, 'FFT_Imin'] = imagFFT.min()



    X.loc[seg_id, 'FFT_Rmean_first_6000'] = realFFT[:6000].mean()

    X.loc[seg_id, 'FFT_Rstd__first_6000'] = realFFT[:6000].std()

    X.loc[seg_id, 'FFT_Rmax_first_6000'] = realFFT[:6000].max()

    X.loc[seg_id, 'FFT_Rmin_first_6000'] = realFFT[:6000].min()

    X.loc[seg_id, 'FFT_Rmean_first_18000'] = realFFT[:18000].mean()

    X.loc[seg_id, 'FFT_Rstd_first_18000'] = realFFT[:18000].std()

    X.loc[seg_id, 'FFT_Rmax_first_18000'] = realFFT[:18000].max()

    X.loc[seg_id, 'FFT_Rmin_first_18000'] = realFFT[:18000].min()



    del xcz

    del zc



    b, a = des_bw_filter_lp(cutoff=2500)

    xc0 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=2500, high=5000)

    xc1 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=5000, high=7500)

    xc2 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=7500, high=10000)

    xc3 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=10000, high=12500)

    xc4 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=12500, high=15000)

    xc5 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=15000, high=17500)

    xc6 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=17500, high=20000)

    xc7 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_hp(cutoff=20000)

    xc8 = sg.lfilter(b, a, xcdm)



    sigs = [xc, pd.Series(xc0), pd.Series(xc1), pd.Series(xc2), pd.Series(xc3),

            pd.Series(xc4), pd.Series(xc5), pd.Series(xc6), pd.Series(xc7), pd.Series(xc8)]



    for i, sig in enumerate(sigs):

        X.loc[seg_id, 'mean_%d' % i] = sig.mean()

        X.loc[seg_id, 'std_%d' % i] = sig.std()

        X.loc[seg_id, 'max_%d' % i] = sig.max()

        X.loc[seg_id, 'min_%d' % i] = sig.min()



        X.loc[seg_id, 'mean_change_abs_%d' % i] = np.mean(np.diff(sig))

        X.loc[seg_id, 'mean_change_rate_%d' % i] = np.mean(np.nonzero((np.diff(sig) / sig[:-1]))[0])

        X.loc[seg_id, 'abs_max_%d' % i] = np.abs(sig).max()

        X.loc[seg_id, 'abs_min_%d' % i] = np.abs(sig).min()



        X.loc[seg_id, 'std_first_50000_%d' % i] = sig[:50000].std()

        X.loc[seg_id, 'std_last_50000_%d' % i] = sig[-50000:].std()

        X.loc[seg_id, 'std_first_10000_%d' % i] = sig[:10000].std()

        X.loc[seg_id, 'std_last_10000_%d' % i] = sig[-10000:].std()



        X.loc[seg_id, 'avg_first_50000_%d' % i] = sig[:50000].mean()

        X.loc[seg_id, 'avg_last_50000_%d' % i] = sig[-50000:].mean()

        X.loc[seg_id, 'avg_first_10000_%d' % i] = sig[:10000].mean()

        X.loc[seg_id, 'avg_last_10000_%d' % i] = sig[-10000:].mean()



        X.loc[seg_id, 'min_first_50000_%d' % i] = sig[:50000].min()

        X.loc[seg_id, 'min_last_50000_%d' % i] = sig[-50000:].min()

        X.loc[seg_id, 'min_first_10000_%d' % i] = sig[:10000].min()

        X.loc[seg_id, 'min_last_10000_%d' % i] = sig[-10000:].min()



        X.loc[seg_id, 'max_first_50000_%d' % i] = sig[:50000].max()

        X.loc[seg_id, 'max_last_50000_%d' % i] = sig[-50000:].max()

        X.loc[seg_id, 'max_first_10000_%d' % i] = sig[:10000].max()

        X.loc[seg_id, 'max_last_10000_%d' % i] = sig[-10000:].max()



        X.loc[seg_id, 'max_to_min_%d' % i] = sig.max() / np.abs(sig.min())

        X.loc[seg_id, 'max_to_min_diff_%d' % i] = sig.max() - np.abs(sig.min())

        X.loc[seg_id, 'count_big_%d' % i] = len(sig[np.abs(sig) > 500])

        X.loc[seg_id, 'sum_%d' % i] = sig.sum()



        X.loc[seg_id, 'mean_change_rate_first_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:50000]) / sig[:50000][:-1]))[0])

        X.loc[seg_id, 'mean_change_rate_last_50000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-50000:]) / sig[-50000:][:-1]))[0])

        X.loc[seg_id, 'mean_change_rate_first_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[:10000]) / sig[:10000][:-1]))[0])

        X.loc[seg_id, 'mean_change_rate_last_10000_%d' % i] = np.mean(np.nonzero((np.diff(sig[-10000:]) / sig[-10000:][:-1]))[0])



        X.loc[seg_id, 'q95_%d' % i] = np.quantile(sig, 0.95)

        X.loc[seg_id, 'q99_%d' % i] = np.quantile(sig, 0.99)

        X.loc[seg_id, 'q05_%d' % i] = np.quantile(sig, 0.05)

        X.loc[seg_id, 'q01_%d' % i] = np.quantile(sig, 0.01)



        X.loc[seg_id, 'abs_q95_%d' % i] = np.quantile(np.abs(sig), 0.95)

        X.loc[seg_id, 'abs_q99_%d' % i] = np.quantile(np.abs(sig), 0.99)

        X.loc[seg_id, 'abs_q05_%d' % i] = np.quantile(np.abs(sig), 0.05)

        X.loc[seg_id, 'abs_q01_%d' % i] = np.quantile(np.abs(sig), 0.01)



        X.loc[seg_id, 'trend_%d' % i] = add_trend_feature(sig)

        X.loc[seg_id, 'abs_trend_%d' % i] = add_trend_feature(sig, abs_values=True)

        X.loc[seg_id, 'abs_mean_%d' % i] = np.abs(sig).mean()

        X.loc[seg_id, 'abs_std_%d' % i] = np.abs(sig).std()



        X.loc[seg_id, 'mad_%d' % i] = sig.mad()

        X.loc[seg_id, 'kurt_%d' % i] = sig.kurtosis()

        X.loc[seg_id, 'skew_%d' % i] = sig.skew()

        X.loc[seg_id, 'med_%d' % i] = sig.median()



        X.loc[seg_id, 'Hilbert_mean_%d' % i] = np.abs(hilbert(sig)).mean()

        X.loc[seg_id, 'Hann_window_mean'] = (convolve(xc, hann(150), mode='same') / sum(hann(150))).mean()



        X.loc[seg_id, 'classic_sta_lta1_mean_%d' % i] = classic_sta_lta(sig, 500, 10000).mean()

        X.loc[seg_id, 'classic_sta_lta2_mean_%d' % i] = classic_sta_lta(sig, 5000, 100000).mean()

        X.loc[seg_id, 'classic_sta_lta3_mean_%d' % i] = classic_sta_lta(sig, 3333, 6666).mean()

        X.loc[seg_id, 'classic_sta_lta4_mean_%d' % i] = classic_sta_lta(sig, 10000, 25000).mean()



        X.loc[seg_id, 'Moving_average_700_mean_%d' % i] = sig.rolling(window=700).mean().mean(skipna=True)

        X.loc[seg_id, 'Moving_average_1500_mean_%d' % i] = sig.rolling(window=1500).mean().mean(skipna=True)

        X.loc[seg_id, 'Moving_average_3000_mean_%d' % i] = sig.rolling(window=3000).mean().mean(skipna=True)

        X.loc[seg_id, 'Moving_average_6000_mean_%d' % i] = sig.rolling(window=6000).mean().mean(skipna=True)



        ewma = pd.Series.ewm

        X.loc[seg_id, 'exp_Moving_average_300_mean_%d' % i] = ewma(sig, span=300).mean().mean(skipna=True)

        X.loc[seg_id, 'exp_Moving_average_3000_mean_%d' % i] = ewma(sig, span=3000).mean().mean(skipna=True)

        X.loc[seg_id, 'exp_Moving_average_30000_mean_%d' % i] = ewma(sig, span=6000).mean().mean(skipna=True)



        no_of_std = 2

        X.loc[seg_id, 'MA_700MA_std_mean_%d' % i] = sig.rolling(window=700).std().mean()

        X.loc[seg_id, 'MA_700MA_BB_high_mean_%d' % i] = (

                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()

        X.loc[seg_id, 'MA_700MA_BB_low_mean_%d' % i] = (

                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_700MA_std_mean_%d' % i]).mean()

        X.loc[seg_id, 'MA_400MA_std_mean_%d' % i] = sig.rolling(window=400).std().mean()

        X.loc[seg_id, 'MA_400MA_BB_high_mean_%d' % i] = (

                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] + no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()

        X.loc[seg_id, 'MA_400MA_BB_low_mean_%d' % i] = (

                    X.loc[seg_id, 'Moving_average_700_mean_%d' % i] - no_of_std * X.loc[seg_id, 'MA_400MA_std_mean_%d' % i]).mean()

        X.loc[seg_id, 'MA_1000MA_std_mean_%d' % i] = sig.rolling(window=1000).std().mean()



        X.loc[seg_id, 'iqr_%d' % i] = np.subtract(*np.percentile(sig, [75, 25]))

        X.loc[seg_id, 'q999_%d' % i] = np.quantile(sig, 0.999)

        X.loc[seg_id, 'q001_%d' % i] = np.quantile(sig, 0.001)

        X.loc[seg_id, 'ave10_%d' % i] = stats.trim_mean(sig, 0.1)



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

        X.loc[seg_id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(

            np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])

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

        X.loc[seg_id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(

            np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])

        X.loc[seg_id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()



    return X
def build_test_fields():

    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, 'train_x.csv'))

    try:

        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)

    except:

        pass



    submission = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'), index_col='seg_id')

    test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)



    print('start for loop')

    count = 0

    for seg_id in tqdm_notebook(test_X.index):  # just tqdm in IDE

        seg = pd.read_csv(os.path.join(DATA_DIR, 'test', str(seg_id) + '.csv'))

        # train_X = create_features_pk_det(seg_id, seg, train_X, start_idx, end_idx)

        test_X = create_features(seg_id, seg, test_X, 0, 0)



        if count % 100 == 0:

            print('working', seg_id)

        count += 1



    test_X.to_csv(os.path.join(OUTPUT_DIR, 'test_x.csv'), index=False)
def scale_fields(fn_train='train_x.csv', fn_test='test_x.csv', 

                 fn_out_train='scaled_train_X.csv' , fn_out_test='scaled_test_X.csv'):

    train_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_train))

    try:

        train_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)

    except:

        pass

    test_X = pd.read_csv(os.path.join(OUTPUT_DIR, fn_test))

    test_X.drop(labels=['seg_id', 'seg_start', 'seg_end'], axis=1, inplace=True)



    print('start scaler')

    scaler = StandardScaler()

    scaler.fit(train_X)

    scaled_train_X = pd.DataFrame(scaler.transform(train_X), columns=train_X.columns)

    scaled_test_X = pd.DataFrame(scaler.transform(test_X), columns=test_X.columns)



    scaled_train_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_train), index=False)

    scaled_test_X.to_csv(os.path.join(OUTPUT_DIR, fn_out_test), index=False)
%%time

build_test_fields()
%%time

scale_fields()