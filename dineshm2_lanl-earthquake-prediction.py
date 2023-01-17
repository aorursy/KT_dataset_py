# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai import *

from fastai.tabular import *
from scipy import stats

from scipy.signal import hann

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import convolve

from sklearn.svm import NuSVR, SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold

warnings.filterwarnings("ignore")
path = Path("../input")
print('There are {} files in test folder '.format(len(os.listdir(os.path.join("../input", 'test' )))))
%%time

df_train = pd.read_csv(path/'train.csv',dtype={'acoustic_data':np.int16,'time_to_failure':np.float32})
df_train.shape
df_train.head(2)
train_acoustic_df = df_train['acoustic_data'][:10]
train_acoustic_df =df_train['acoustic_data'].values[::100]

train_acoustic_df.shape
train_time_to_failure_df = df_train['time_to_failure'].values[::100]
%matplotlib inline

import matplotlib
def plot_acc_ttf_data(train_acoustic_df,train_time_to_failure_df,title='Acoustic data and time to failure: sampled 1%'):

    fig ,ax1 = plt.subplots(figsize=(12,9))

    plt.title(title)

    plt.plot(train_acoustic_df,color='r')

    ax1.set_ylabel('train_acoustic_df',color='r')

    plt.legend(['acoustic-data'],loc=(0.01,0.95))

    ax2 = ax1.twinx()

    plt.plot(train_time_to_failure_df,color='b')

    ax2.set_ylabel('time to failure',color='b')

    plt.legend(['time_to_failure'],loc=(0.01,0.9))

    plt.grid(True)

    

    
plot_acc_ttf_data(train_acoustic_df,train_time_to_failure_df)
del train_time_to_failure_df

del train_acoustic_df
train_acoustic_df = df_train['acoustic_data'].values[:6291455]

train_time_to_failure_df = df_train['time_to_failure'].values[:6291455]
plot_acc_ttf_data(train_acoustic_df,train_time_to_failure_df)
rows=150000



df_train.shape



segments = int(np.floor(df_train.shape[0]/rows))

print("number of segments ",segments)
train_X = pd.DataFrame(index=range(segments),dtype=np.float64)

train_y= pd.DataFrame(index=range(segments),dtype=np.float64,columns=['time_to_failure'])
total_mean= df_train['acoustic_data'].mean()

total_std= df_train['acoustic_data'].std()

total_max= df_train['acoustic_data'].max()

total_min= df_train['acoustic_data'].min()

total_sum= df_train['acoustic_data'].sum()

total_abs_sum= np.abs(df_train['acoustic_data']).sum()
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

    

    X.loc[seg_id, 'trend'] = add_trend_feature(xc)

    X.loc[seg_id, 'abs_trend'] = add_trend_feature(xc, abs_values=True)

    X.loc[seg_id, 'abs_mean'] = np.abs(xc).mean()

    X.loc[seg_id, 'abs_std'] = np.abs(xc).std()

    

    X.loc[seg_id, 'mad'] = xc.mad()

    X.loc[seg_id, 'kurt'] = xc.kurtosis()

    X.loc[seg_id, 'skew'] = xc.skew()

    

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

    seg = df_train.iloc[seg_id*rows:seg_id*rows+rows]

    create_features(seg_id, seg, train_X)

    train_y.loc[seg_id, 'time_to_failure'] = seg['time_to_failure'].values[-1]

df = pd.concat([train_X, train_y], axis=1)
dep_var = 'time_to_failure'

cont_names = None

cat_names = None



procs = [FillMissing, Categorify, Normalize]

submission = pd.read_csv(path/'sample_submission.csv', index_col='seg_id')

test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
for seg_id in tqdm(test_X.index):

    seg = pd.read_csv(path/f'test/{seg_id}.csv')

    create_features(seg_id, seg, test_X)
test_X.reset_index(inplace=True)
df = pd.concat([train_X, train_y], axis=1)
df.to_feather("df")

test_X.to_hdf('df_test.h5', key="df_test")

df = pd.read_feather("df")

test_df = pd.read_hdf('df_test.h5', key="df_test")
dep_var = 'time_to_failure'

cont_names = None

cat_names = None



procs = [FillMissing, Categorify, Normalize]
data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=range(0,800), procs=procs, test_df=test_df)
data.show_batch(rows=10)
learn = tabular_learner(data, layers=[200,100], model_dir=Path("/"))
learn.lr_find(); learn.recorder.plot()

lr = 1e-02
learn.fit_one_cycle(5, slice(lr))
learn.save('stage-1')
learn.load('stage-1');
learn.unfreeze()
learn.lr_find(); learn.recorder.plot()
learn.fit_one_cycle(10, slice(2e-05))
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
subm_df = pd.DataFrame(test_df["seg_id"])
subm_df["time_to_failure"] = preds.numpy()
subm_df.head()
subm_df.to_csv('submission.csv', index=False)