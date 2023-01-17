# Setting package umum 
import pandas as pd
import pandas_profiling as pp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tqdm import tqdm_notebook as tqdm
%matplotlib inline

from matplotlib.pylab import rcParams
# For every plotting cell use this
rcParams['figure.figsize'] = [10,5]
plt.style.use('fivethirtyeight') 
sns.set_style('whitegrid')
# fig, axes = plt.subplots()
# grid = gridspec.GridSpec(n_row,n_col)
# ax = plt.subplot(grid[i])

import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.options.display.float_format = '{:.4f}'.format

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Load dataset
df_train_clean = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv')
df_test_clean = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv')
df_train_kalman = pd.read_csv('/kaggle/input/data-without-drift-with-kalman-filter/train.csv')
df_test_kalman = pd.read_csv('/kaggle/input/data-without-drift-with-kalman-filter/test.csv')
df_train_rfc = pd.DataFrame(np.load('/kaggle/input/ion-shifted-rfc-proba/Y_train_proba.npy'))
df_test_rfc = pd.DataFrame(np.load('/kaggle/input/ion-shifted-rfc-proba/Y_test_proba.npy'))
# Add `kalman` feature
df_train_clean['kalman'] = df_train_kalman['signal']
df_test_clean['kalman'] = df_test_kalman['signal']
# Add `shifted_prob_class` feature
list_shifted_var = []
for clas in list(df_train_rfc.columns) :
    var_name = 'shited_proba_class' + str(clas)
    df_train_clean[var_name] = df_train_rfc[clas]
    df_test_clean[var_name] = df_test_rfc[clas]
    list_shifted_var.append(var_name)
### Make `previous_signal`
list_prev_var = []
def make_previous_signal(df, list_ts) :
    
    # Iterate to make signal
    for ts in list_ts :
        
        # Make the signal
        var_name = 'previous_signal_t'+ str(ts)
        df[var_name] = df['signal'].shift(ts)
        
        # Fill missin values with mean
        row_nan = list(range(ts))
        df.loc[row_nan, var_name] = [np.mean(df.loc[row:ts+2, 'signal']) for row in row_nan]
        
        list_prev_var.append(var_name)
        
    return df
        
df_train_clean = make_previous_signal(df_train_clean, [1,3,5,50,100])
df_test_clean = make_previous_signal(df_test_clean, [1,3,5,50,100])

list_prev_var = list(set(list_prev_var))
### Make `future_signal`
list_fut_var = []
def make_future_signal(df, list_ts) :
    
    # Iterate to make signal
    for ts in list_ts :
        
        # Make the signal
        var_name = 'future_signal_t'+ str(ts)
        df[var_name] = df['signal'].shift(-ts, fill_value=np.mean(df['signal']))
        
        list_fut_var.append(var_name)
        
    return df
        
df_train_clean = make_future_signal(df_train_clean, [1,3,5,50,100])
df_test_clean = make_future_signal(df_test_clean, [1,3,5,50,100])

list_fut_var = list(set(list_fut_var))
### Make 'moving average' :
list_ma_var = []
def moving_average(df, list_ts) :
    
    # Iterate to make MA
    for ts in list_ts :
        
        # Make MA
        var_name = 'ma_t' + str(ts)
        df[var_name] = df['signal'].rolling(ts).mean()
        
        # Fill missing values with mean
        row_nan = list(range(ts))
        df.loc[row_nan, var_name] = [np.mean(df.loc[i+ts:2*(ts)+i, var_name]) for i in range(ts)]
        
        list_ma_var.append(var_name)
        
    return df

df_train_clean = moving_average(df_train_clean, [100,300,500])
df_test_clean = moving_average(df_test_clean, [100,300,500])

list_ma_var = list(set(list_ma_var))
### Make `signal_power`
df_train_clean['signal_power'] = df_train_clean['signal'] ** 2
df_test_clean['signal_power'] = df_test_clean['signal'] ** 2
### Make batch data train
train_batch = int(len(df_train_clean) / 10)
train_group = [1,1,1,2,4,3,1,2,3,4]
df_train_clean['group'] = np.nan

for i,group in enumerate(train_group) :
    df_train_clean.iloc[i*train_batch : (i+1)*train_batch, -1] = group
        
### Make batch data test
test_batch = int(len(df_test_clean) / 20)
test_group = [1,2,3,1,1,4,3,4,1,2,1,1,1,1,1,1,1,1,1,1]
df_test_clean['group'] = np.nan

for i,group in enumerate(test_group) :
    df_test_clean.iloc[i*test_batch : (i+1)*test_batch, -1] = group
### Make `signal_deviate`
train_batch = int(len(df_train_clean) / 10)
df_train_clean['signal_deviate'] = np.nan

for i,group in enumerate(train_group) :
    df_train_clean.iloc[i*train_batch : (i+1)*train_batch, -1] = df_train_clean['signal_power'] - np.mean(df_train_clean.iloc[i*train_batch : (i+1)*train_batch, -3])
    
test_batch = int(len(df_test_clean) / 20)
df_test_clean['signal_deviate'] = np.nan

for i,group in enumerate(test_group) :
    df_test_clean.iloc[i*test_batch : (i+1)*test_batch, -1] = df_test_clean['signal_power'] - np.mean(df_test_clean.iloc[i*test_batch : (i+1)*test_batch, -3])
### Make `weight`
dict_channels = (1 - (df_train_clean['open_channels'].value_counts() / len(df_train_clean))).to_dict()
df_train_clean['weight'] = df_train_clean['open_channels'].map(dict_channels)
### Make 'Tarun_wavelet'
import pywt

def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise_signal(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')

df_train_clean['Tarun_wavelet'] = denoise_signal(df_train_clean['signal'])
df_test_clean['Tarun_wavelet'] = denoise_signal(df_test_clean['signal'])
### Make 'MProx_wavelet'
import pywt

def denoise_signal_mprox(x, wavelet='sym4') :
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(x), w.dec_len)
    threshold = 0.04
    
    coeff = pywt.wavedec(x, 'sym4', level=maxlev)
    coeff[1:] = (pywt.threshold(i, value=threshold*max(i)) for i in coeff[1:])
    
    return pywt.waverec(coeff, wavelet)

df_train_clean['MProx_wavelet'] = denoise_signal_mprox(df_train_clean['signal'])
df_test_clean['MProx_wavelet'] = denoise_signal_mprox(df_test_clean['signal'])
### Make 'Tarun_perm_ent'
def _embed(x, order=3, delay=1):
    N = len(x)
    if order * delay > N:
        raise ValueError("Error: order * delay should be lower than x.size")
    if delay < 1:
        raise ValueError("Delay has to be at least 1.")
    if order < 2:
        raise ValueError("Order has to be at least 2.")
    Y = np.zeros((order, N - (order - 1) * delay))
    for i in range(order):
        Y[i] = x[i * delay:i * delay + Y.shape[1]]
    return Y.T

all = ['perm_entropy', 'spectral_entropy', 'svd_entropy', 'app_entropy',
       'sample_entropy']


def perm_entropy(x, order=3, delay=1, normalize=False):
    x = np.array(x)
    ran_order = range(order)
    hashmult = np.power(order, ran_order)
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe

def make_perm_entropy(df) :
    pe = [np.nan]*100
    for i in range(100,len(df)) :
        pe.append(perm_entropy(df.loc[i-100 : i, 'signal']))
        
    return pe

df_train_clean['Tarun_perm_ent'] = make_perm_entropy(df_train_clean)
df_train_clean['Tarun_perm_ent'] = df_train_clean['Tarun_perm_ent'].fillna(np.mean(df_train_clean['Tarun_perm_ent']))
df_test_clean['Tarun_perm_ent'] = make_perm_entropy(df_test_clean)
df_test_clean['Tarun_perm_ent'] = df_test_clean['Tarun_perm_ent'].fillna(np.mean(df_test_clean['Tarun_perm_ent']))
### Make 'Tarun_apx_ent'
from sklearn.neighbors import KDTree

def _app_samp_entropy(x, order, metric='chebyshev', approximate=True):
    """Utility function for `app_entropy`` and `sample_entropy`.
    """
    _all_metrics = KDTree.valid_metrics
    if metric not in _all_metrics:
        raise ValueError('The given metric (%s) is not valid. The valid '
                         'metric names are: %s' % (metric, _all_metrics))
    phi = np.zeros(2)
    r = 0.2 * np.std(x, ddof=1)

    # compute phi(order, r)
    _emb_data1 = _embed(x, order, 1)
    if approximate:
        emb_data1 = _emb_data1
    else:
        emb_data1 = _emb_data1[:-1]
    count1 = KDTree(emb_data1, metric=metric).query_radius(emb_data1, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    # compute phi(order + 1, r)
    emb_data2 = _embed(x, order + 1, 1)
    count2 = KDTree(emb_data2, metric=metric).query_radius(emb_data2, r,
                                                           count_only=True
                                                           ).astype(np.float64)
    if approximate:
        phi[0] = np.mean(np.log(count1 / emb_data1.shape[0]))
        phi[1] = np.mean(np.log(count2 / emb_data2.shape[0]))
    else:
        phi[0] = np.mean((count1 - 1) / (emb_data1.shape[0] - 1))
        phi[1] = np.mean((count2 - 1) / (emb_data2.shape[0] - 1))
    return phi


def _numba_sampen(x, mm=2, r=0.2):
    """
    Fast evaluation of the sample entropy using Numba.
    """
    n = x.size
    n1 = n - 1
    mm += 1
    mm_dbld = 2 * mm

    # Define threshold
    r *= x.std()

    # initialize the lists
    run = [0] * n
    run1 = run[:]
    r1 = [0] * (n * mm_dbld)
    a = [0] * mm
    b = a[:]
    p = a[:]

    for i in range(n1):
        nj = n1 - i

        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - x[i]) < r:
                run[jj] = run1[jj] + 1
                m1 = mm if mm < run[jj] else run[jj]
                for m in range(m1):
                    a[m] += 1
                    if j < n1:
                        b[m] += 1
            else:
                run[jj] = 0
        for j in range(mm_dbld):
            run1[j] = run[j]
            r1[i + n * j] = run[j]
        if nj > mm_dbld - 1:
            for j in range(mm_dbld, nj):
                run1[j] = run[j]

    m = mm - 1

    while m > 0:
        b[m] = b[m - 1]
        m -= 1

    b[0] = n * n1 / 2
    a = np.array([float(aa) for aa in a])
    b = np.array([float(bb) for bb in b])
    p = np.true_divide(a, b)
    return -log(p[-1])


def app_entropy(x, order=2, metric='chebyshev'):
    phi = _app_samp_entropy(x, order=order, metric=metric, approximate=True)
    return np.subtract(phi[0], phi[1])

def make_apx_entropy(df) :
    ae = [np.nan]*100
    for i in range(100,len(df)) :
        ae.append(app_entropy(df.loc[i-100 : i, 'signal']))
        
    return ae

df_train_clean['Tarun_apx_ent'] = make_apx_entropy(df_train_clean)
df_train_clean['Tarun_apx_ent'] = df_train_clean['Tarun_apx_ent'].fillna(np.mean(df_train_clean['Tarun_apx_ent']))
df_test_clean['Tarun_apx_ent'] = make_apx_entropy(df_test_clean)
df_test_clean['Tarun_apx_ent'] = df_test_clean['Tarun_apx_ent'].fillna(np.mean(df_test_clean['Tarun_apx_ent']))
### Make 'Tarun_higuchi'
from math import log, floor

def _log_n(min_n, max_n, factor):
    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))
    ns = [min_n]
    for i in range(max_i + 1):
        n = int(floor(min_n * (factor ** i)))
        if n > ns[-1]:
            ns.append(n)
    return np.array(ns, dtype=np.int64)

def _higuchi_fd(x, kmax):
    n_times = x.size
    lk = np.empty(kmax)
    x_reg = np.empty(kmax)
    y_reg = np.empty(kmax)
    for k in range(1, kmax + 1):
        lm = np.empty((k,))
        for m in range(k):
            ll = 0
            n_max = floor((n_times - m - 1) / k)
            n_max = int(n_max)
            for j in range(1, n_max):
                ll += abs(x[m + j * k] - x[m + (j - 1) * k])
            ll /= k
            ll *= (n_times - 1) / (k * n_max)
            lm[m] = ll
        # Mean of lm
        m_lm = 0
        for m in range(k):
            m_lm += lm[m]
        m_lm /= k
        lk[k - 1] = m_lm
        x_reg[k - 1] = log(1. / k)
        y_reg[k - 1] = log(m_lm)
    higuchi, _ = _linear_regression(x_reg, y_reg)
    return higuchi


def higuchi_fd(x, kmax=10):
    x = np.asarray(x, dtype=np.float64)
    kmax = int(kmax)
    return _higuchi_fd(x, kmax)

def _linear_regression(x, y):
    n_times = x.size
    sx2 = 0
    sx = 0
    sy = 0
    sxy = 0
    for j in range(n_times):
        sx2 += x[j] ** 2
        sx += x[j]
        sxy += x[j] * y[j]
        sy += y[j]
    den = n_times * sx2 - (sx ** 2)
    num = n_times * sxy - sx * sy
    slope = num / den
    intercept = np.mean(y) - slope * np.mean(x)
    return slope, intercept

def make_higuchi(df) :
    h = [np.nan]*100
    for i in range(100,len(df)) :
        h.append(higuchi_fd(df.loc[i-100 : i, 'signal']))
        
    return h

df_train_clean['Tarun_higuchi'] = make_higuchi(df_train_clean)
df_train_clean['Tarun_higuchi'] = df_train_clean['Tarun_higuchi'].fillna(np.mean(df_train_clean['Tarun_higuchi']))
df_test_clean['Tarun_higuchi'] = make_higuchi(df_test_clean)
df_test_clean['Tarun_higuchi'] = df_test_clean['Tarun_higuchi'].fillna(np.mean(df_test_clean['Tarun_higuchi']))
### Make 'Tarun_katz'
def katz_fd(x):
    x = np.array(x)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))

def make_katz(df) :
    k = [np.nan]*100
    for i in range(100,len(df)) :
        k.append(katz_fd(df.loc[i-100 : i, 'signal']))
        
    return k

df_train_clean['Tarun_katz'] = make_katz(df_train_clean)
df_train_clean['Tarun_katz'] = df_train_clean['Tarun_katz'].fillna(np.mean(df_train_clean['Tarun_katz']))
df_test_clean['Tarun_katz'] = make_katz(df_test_clean)
df_test_clean['Tarun_katz'] = df_test_clean['Tarun_katz'].fillna(np.mean(df_test_clean['Tarun_katz']))
# Save the dataset
df_train_clean.to_csv('data_train_hope.csv' ,index=False)
df_test_clean.to_csv('data_test_hope.csv' ,index=False)