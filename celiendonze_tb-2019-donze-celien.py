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
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

import warnings

import sys

import gc



from scipy.stats import kurtosis

from scipy.stats import skew

from scipy.stats import kstat

from scipy.signal import hilbert

from scipy.signal import hann

from scipy.signal import convolve

from scipy.signal import welch



import scipy.signal as sg



from tsfresh.feature_extraction import feature_calculators



import nolds

import librosa



from imblearn.over_sampling import SMOTE



from catboost import CatBoostRegressor, Pool

import tensorflow

from keras.models import Sequential

from keras.layers import Dense, LSTM

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler

from sklearn.svm import NuSVR

from sklearn.svm import SVR

from sklearn.manifold import TSNE

from sklearn.metrics import mean_absolute_error, SCORERS

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLasso, ElasticNet, MultiTaskElasticNet, Ridge

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
%matplotlib inline

%precision 4

warnings.filterwarnings('ignore')

plt.style.use('ggplot')

np.set_printoptions(suppress=True)

pd.set_option("display.precision", 15)
print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))

print('Tensorflow: {}'.format(tensorflow.__version__))
def load_data(nrows):

    """load the data for exploration"""

    filename = "../input/train.csv"

    

    return pd.read_csv(

        filename,

        dtype={

            'acoustic_data': np.int16,

            'time_to_failure': np.float32

        },

        nrows=nrows,

        skiprows=1,

        names = ['acoustic_data', 'time_to_failure']

    )





def get_data_iterator(nrows):

    return pd.read_csv('../input/train.csv',

        iterator=True,

        dtype={

            'acoustic_data': np.int16,

            'time_to_failure': np.float32

        },

        nrows=nrows,

        skiprows=1,

        names = ['acoustic_data', 'time_to_failure']

    )

%%time

nrows = 600_000_000

data_df = load_data(nrows)
data_df.head()
data_df.shape
acoustic_data_small = data_df['acoustic_data'].values[::50]

time_to_failure_small = data_df['time_to_failure'].values[::50]



fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")

plt.plot(acoustic_data_small, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(time_to_failure_small, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)



del acoustic_data_small

del time_to_failure_small

gc.collect()
acoustic_data_small = data_df['acoustic_data'].values[:150_000:50]

time_to_failure_small = data_df['time_to_failure'].values[:150_000:50]



fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time_to_failure. first 150_000, 2% of data (sampled)")

plt.plot(acoustic_data_small, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(time_to_failure_small, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

#plt.grid(False)



del acoustic_data_small

del time_to_failure_small

gc.collect()
sig = data_df['acoustic_data'].values[:150_000]

sig = sig - np.mean(sig)

fft = np.fft.rfft(sig)

plt.plot(fft)

plt.title("FFT of first 150_000, without the mean")

plt.show()
plt.plot(np.abs(fft))

plt.title("FFT magnitude of first 150_000, without the mean")

plt.show()
plt.plot(np.angle(fft))

plt.title("FFT phase of first 150_000, without the mean")

plt.show()
hilbertSignal = hilbert(sig)

plt.plot(hilbertSignal)

plt.title("Hilbert signal")

plt.show()
plt.plot(np.abs(hilbertSignal))

plt.title("Hilbert signal, magnitude")

plt.show()
plt.plot(np.angle(hilbertSignal))

plt.title("Hilbert signal, phase")

plt.show()
del data_df

gc.collect()
NY_FREQ_IDX = 75000  # the test signals are 150k samples long, Nyquist is thus 75k.

CUTOFF = 18000

MAX_FREQ_IDX = 20000

FREQ_STEP = 2500
def des_bw_filter_lp(cutoff=CUTOFF):  # low pass filter

    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX)

    return b, a



def des_bw_filter_hp(cutoff=CUTOFF):  # high pass filter

    b, a = sg.butter(4, Wn=cutoff/NY_FREQ_IDX, btype='highpass')

    return b, a



def des_bw_filter_bp(low, high):  # band pass filter

    b, a = sg.butter(4, Wn=(low/NY_FREQ_IDX, high/NY_FREQ_IDX), btype='bandpass')

    return b, a
def add_trend_feature(arr, abs_values=False):

    arr = arr[::50]

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
def calc_change_rate(x):

    change = (np.diff(x) / x[:-1]).values

    change = change[np.nonzero(change)[0]]

    change = change[~np.isnan(change)]

    change = change[change != -np.inf]

    change = change[change != np.inf]

    return np.mean(change)
from numba import jit

from math import log, floor



def _embed(x, order=3, delay=1):

    """Time-delay embedding.

    Parameters

    ----------

    x : 1d-array, shape (n_times)

        Time series

    order : int

        Embedding dimension (order)

    delay : int

        Delay.

    Returns

    -------

    embedded : ndarray, shape (n_times - (order - 1) * delay, order)

        Embedded time-series.

    """

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





@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)

def _linear_regression(x, y):

    """Fast linear regression using Numba.

    Parameters

    ----------

    x, y : ndarray, shape (n_times,)

        Variables

    Returns

    -------

    slope : float

        Slope of 1D least-square regression.

    intercept : float

        Intercept

    """

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





@jit('i8[:](f8, f8, f8)', nopython=True)

def _log_n(min_n, max_n, factor):

    """

    Creates a list of integer values by successively multiplying a minimum

    value min_n by a factor > 1 until a maximum value max_n is reached.

    Used for detrended fluctuation analysis (DFA).

    Function taken from the nolds python package

    (https://github.com/CSchoel/nolds) by Christopher Scholzel.

    Parameters

    ----------

    min_n (float):

        minimum value (must be < max_n)

    max_n (float):

        maximum value (must be > min_n)

    factor (float):

       factor used to increase min_n (must be > 1)

    Returns

    -------

    list of integers:

        min_n, min_n * factor, min_n * factor^2, ... min_n * factor^i < max_n

        without duplicates

    """

    max_i = int(floor(log(1.0 * max_n / min_n) / log(factor)))

    ns = [min_n]

    for i in range(max_i + 1):

        n = int(floor(min_n * (factor ** i)))

        if n > ns[-1]:

            ns.append(n)

    return np.array(ns, dtype=np.int64)



from sklearn.neighbors import KDTree

from scipy.signal import periodogram, welch



def perm_entropy(x, order=3, delay=1, normalize=False):

    """Permutation Entropy.

    Parameters

    ----------

    x : list or np.array

        One-dimensional time series of shape (n_times)

    order : int

        Order of permutation entropy

    delay : int

        Time delay

    normalize : bool

        If True, divide by log2(order!) to normalize the entropy between 0

        and 1. Otherwise, return the permutation entropy in bit.

    Returns

    -------

    pe : float

        Permutation Entropy

    Notes

    -----

    The permutation entropy is a complexity measure for time-series first

    introduced by Bandt and Pompe in 2002 [1]_.

    The permutation entropy of a signal :math:`x` is defined as:

    .. math:: H = -\\sum p(\\pi)log_2(\\pi)

    where the sum runs over all :math:`n!` permutations :math:`\\pi` of order

    :math:`n`. This is the information contained in comparing :math:`n`

    consecutive values of the time series. It is clear that

    :math:`0 ≤ H (n) ≤ log_2(n!)` where the lower bound is attained for an

    increasing or decreasing sequence of values, and the upper bound for a

    completely random system where all :math:`n!` possible permutations appear

    with the same probability.

    The embedded matrix :math:`Y` is created by:

    .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]

    .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T

    References

    ----------

    .. [1] Bandt, Christoph, and Bernd Pompe. "Permutation entropy: a

           natural complexity measure for time series." Physical review letters

           88.17 (2002): 174102.

    Examples

    --------

    1. Permutation entropy with order 2

        >>> from entropy import perm_entropy

        >>> x = [4, 7, 9, 10, 6, 11, 3]

        >>> # Return a value in bit between 0 and log2(factorial(order))

        >>> print(perm_entropy(x, order=2))

            0.918

    2. Normalized permutation entropy with order 3

        >>> from entropy import perm_entropy

        >>> x = [4, 7, 9, 10, 6, 11, 3]

        >>> # Return a value comprised between 0 and 1.

        >>> print(perm_entropy(x, order=3, normalize=True))

            0.589

    """

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





def spectral_entropy(x, sf, method='fft', nperseg=None, normalize=False):

    """Spectral Entropy.

    Parameters

    ----------

    x : list or np.array

        One-dimensional time series of shape (n_times)

    sf : float

        Sampling frequency

    method : str

        Spectral estimation method ::

        'fft' : Fourier Transform (via scipy.signal.periodogram)

        'welch' : Welch periodogram (via scipy.signal.welch)

    nperseg : str or int

        Length of each FFT segment for Welch method.

        If None, uses scipy default of 256 samples.

    normalize : bool

        If True, divide by log2(psd.size) to normalize the spectral entropy

        between 0 and 1. Otherwise, return the spectral entropy in bit.

    Returns

    -------

    se : float

        Spectral Entropy

    Notes

    -----

    Spectral Entropy is defined to be the Shannon Entropy of the Power

    Spectral Density (PSD) of the data:

    .. math:: H(x, sf) =  -\\sum_{f=0}^{f_s/2} PSD(f) log_2[PSD(f)]

    Where :math:`PSD` is the normalised PSD, and :math:`f_s` is the sampling

    frequency.

    References

    ----------

    .. [1] Inouye, T. et al. (1991). Quantification of EEG irregularity by

       use of the entropy of the power spectrum. Electroencephalography

       and clinical neurophysiology, 79(3), 204-210.

    Examples

    --------

    1. Spectral entropy of a pure sine using FFT

        >>> from entropy import spectral_entropy

        >>> import numpy as np

        >>> sf, f, dur = 100, 1, 4

        >>> N = sf * duration # Total number of discrete samples

        >>> t = np.arange(N) / sf # Time vector

        >>> x = np.sin(2 * np.pi * f * t)

        >>> print(np.round(spectral_entropy(x, sf, method='fft'), 2)

            0.0

    2. Spectral entropy of a random signal using Welch's method

        >>> from entropy import spectral_entropy

        >>> import numpy as np

        >>> np.random.seed(42)

        >>> x = np.random.rand(3000)

        >>> print(spectral_entropy(x, sf=100, method='welch'))

            9.939

    3. Normalized spectral entropy

        >>> print(spectral_entropy(x, sf=100, method='welch', normalize=True))

            0.995

    """

    x = np.array(x)

    # Compute and normalize power spectrum

    if method == 'fft':

        _, psd = periodogram(x, sf)

    elif method == 'welch':

        _, psd = welch(x, sf, nperseg=nperseg)

    psd_norm = np.divide(psd, psd.sum())

    se = -np.multiply(psd_norm, np.log2(psd_norm)).sum()

    if normalize:

        se /= np.log2(psd_norm.size)

    return se





def svd_entropy(x, order=3, delay=1, normalize=False):

    """Singular Value Decomposition entropy.

    Parameters

    ----------

    x : list or np.array

        One-dimensional time series of shape (n_times)

    order : int

        Order of permutation entropy

    delay : int

        Time delay

    normalize : bool

        If True, divide by log2(order!) to normalize the entropy between 0

        and 1. Otherwise, return the permutation entropy in bit.

    Returns

    -------

    svd_e : float

        SVD Entropy

    Notes

    -----

    SVD entropy is an indicator of the number of eigenvectors that are needed

    for an adequate explanation of the data set. In other words, it measures

    the dimensionality of the data.

    The SVD entropy of a signal :math:`x` is defined as:

    .. math::

        H = -\\sum_{i=1}^{M} \\overline{\\sigma}_i log_2(\\overline{\\sigma}_i)

    where :math:`M` is the number of singular values of the embedded matrix

    :math:`Y` and :math:`\\sigma_1, \\sigma_2, ..., \\sigma_M` are the

    normalized singular values of :math:`Y`.

    The embedded matrix :math:`Y` is created by:

    .. math:: y(i)=[x_i,x_{i+delay}, ...,x_{i+(order-1) * delay}]

    .. math:: Y=[y(1),y(2),...,y(N-(order-1))*delay)]^T

    Examples

    --------

    1. SVD entropy with order 2

        >>> from entropy import svd_entropy

        >>> x = [4, 7, 9, 10, 6, 11, 3]

        >>> # Return a value in bit between 0 and log2(factorial(order))

        >>> print(svd_entropy(x, order=2))

            0.762

    2. Normalized SVD entropy with order 3

        >>> from entropy import svd_entropy

        >>> x = [4, 7, 9, 10, 6, 11, 3]

        >>> # Return a value comprised between 0 and 1.

        >>> print(svd_entropy(x, order=3, normalize=True))

            0.687

    """

    x = np.array(x)

    mat = _embed(x, order=order, delay=delay)

    W = np.linalg.svd(mat, compute_uv=False)

    # Normalize the singular values

    W /= sum(W)

    svd_e = -np.multiply(W, np.log2(W)).sum()

    if normalize:

        svd_e /= np.log2(order)

    return svd_e
@jit('UniTuple(float64, 2)(float64[:], float64[:])', nopython=True)

def _linear_regression(x, y):

    """Fast linear regression using Numba.

    Parameters

    ----------

    x, y : ndarray, shape (n_times,)

        Variables

    Returns

    -------

    slope : float

        Slope of 1D least-square regression.

    intercept : float

        Intercept

    """

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



def petrosian_fd(x):

    """Petrosian fractal dimension.

    Parameters

    ----------

    x : list or np.array

        One dimensional time series

    Returns

    -------

    pfd : float

        Petrosian fractal dimension

    Notes

    -----

    The Petrosian algorithm can be used to provide a fast computation of

    the FD of a signal by translating the series into a binary sequence.

    The Petrosian fractal dimension of a time series :math:`x` is defined by:

    .. math:: \\frac{log_{10}(N)}{log_{10}(N) +

       log_{10}(\\frac{N}{N+0.4N_{\\Delta}})}

    where :math:`N` is the length of the time series, and

    :math:`N_{\\Delta}` is the number of sign changes in the binary sequence.

    Original code from the pyrem package by Quentin Geissmann.

    References

    ----------

    .. [1] A. Petrosian, Kolmogorov complexity of finite sequences and

       recognition of different preictal EEG patterns, in , Proceedings of the

       Eighth IEEE Symposium on Computer-Based Medical Systems, 1995,

       pp. 212-217.

    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for

       the computation of EEG biomarkers for dementia." 2nd International

       Conference on Computational Intelligence in Medicine and Healthcare

       (CIMED2005). 2005.

    Examples

    --------

    Petrosian fractal dimension.

        >>> import numpy as np

        >>> from entropy import petrosian_fd

        >>> np.random.seed(123)

        >>> x = np.random.rand(100)

        >>> print(petrosian_fd(x))

            1.0505

    """

    n = len(x)

    # Number of sign changes in the first derivative of the signal

    diff = np.ediff1d(x)

    N_delta = (diff[1:-1] * diff[0:-2] < 0).sum()

    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))





def katz_fd(x):

    """Katz Fractal Dimension.

    Parameters

    ----------

    x : list or np.array

        One dimensional time series

    Returns

    -------

    kfd : float

        Katz fractal dimension

    Notes

    -----

    The Katz Fractal dimension is defined by:

    .. math:: FD_{Katz} = \\frac{log_{10}(n)}{log_{10}(d/L)+log_{10}(n)}

    where :math:`L` is the total length of the time series and :math:`d`

    is the Euclidean distance between the first point in the

    series and the point that provides the furthest distance

    with respect to the first point.

    Original code from the mne-features package by Jean-Baptiste Schiratti

    and Alexandre Gramfort.

    References

    ----------

    .. [1] Esteller, R. et al. (2001). A comparison of waveform fractal

           dimension algorithms. IEEE Transactions on Circuits and Systems I:

           Fundamental Theory and Applications, 48(2), 177-183.

    .. [2] Goh, Cindy, et al. "Comparison of fractal dimension algorithms for

           the computation of EEG biomarkers for dementia." 2nd International

           Conference on Computational Intelligence in Medicine and Healthcare

           (CIMED2005). 2005.

    Examples

    --------

    Katz fractal dimension.

        >>> import numpy as np

        >>> from entropy import katz_fd

        >>> np.random.seed(123)

        >>> x = np.random.rand(100)

        >>> print(katz_fd(x))

            5.1214

    """

    x = np.array(x)

    dists = np.abs(np.ediff1d(x))

    ll = dists.sum()

    ln = np.log10(np.divide(ll, dists.mean()))

    aux_d = x - x[0]

    d = np.max(np.abs(aux_d[1:]))

    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))





@jit('float64(float64[:], int32)')

def _higuchi_fd(x, kmax):

    """Utility function for `higuchi_fd`.

    """

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

    """Higuchi Fractal Dimension.

    Parameters

    ----------

    x : list or np.array

        One dimensional time series

    kmax : int

        Maximum delay/offset (in number of samples).

    Returns

    -------

    hfd : float

        Higuchi Fractal Dimension

    Notes

    -----

    Original code from the mne-features package by Jean-Baptiste Schiratti

    and Alexandre Gramfort.

    The `higuchi_fd` function uses Numba to speed up the computation.

    References

    ----------

    .. [1] Higuchi, Tomoyuki. "Approach to an irregular time series on the

       basis of the fractal theory." Physica D: Nonlinear Phenomena 31.2

       (1988): 277-283.

    Examples

    --------

    Higuchi Fractal Dimension

        >>> import numpy as np

        >>> from entropy import higuchi_fd

        >>> np.random.seed(123)

        >>> x = np.random.rand(100)

        >>> print(higuchi_fd(x))

            2.051179

    """

    x = np.asarray(x, dtype=np.float64)

    kmax = int(kmax)

    return _higuchi_fd(x, kmax)


@jit('f8(f8[:])', nopython=True)

def _dfa(x):

    """

    Utility function for detrended fluctuation analysis

    """

    N = len(x)

    nvals = _log_n(4, 0.1 * N, 1.2)

    walk = np.cumsum(x - x.mean())

    fluctuations = np.zeros(len(nvals))



    for i_n, n in enumerate(nvals):

        d = np.reshape(walk[:N - (N % n)], (N // n, n))

        ran_n = np.array([float(na) for na in range(n)])

        d_len = len(d)

        slope = np.empty(d_len)

        intercept = np.empty(d_len)

        trend = np.empty((d_len, ran_n.size))

        for i in range(d_len):

            slope[i], intercept[i] = _linear_regression(ran_n, d[i])

            y = np.zeros_like(ran_n)

            # Equivalent to np.polyval function

            for p in [slope[i], intercept[i]]:

                y = y * ran_n + p

            trend[i, :] = y

        # calculate standard deviation (fluctuation) of walks in d around trend

        flucs = np.sqrt(np.sum((d - trend) ** 2, axis=1) / n)

        # calculate mean fluctuation over all subsequences

        fluctuations[i_n] = flucs.sum() / flucs.size



    # Filter zero

    nonzero = np.nonzero(fluctuations)[0]

    fluctuations = fluctuations[nonzero]

    nvals = nvals[nonzero]

    if len(fluctuations) == 0:

        # all fluctuations are zero => we cannot fit a line

        dfa = np.nan

    else:

        dfa, _ = _linear_regression(np.log(nvals), np.log(fluctuations))

    return dfa





def detrended_fluctuation(x):

    """

    Detrended fluctuation analysis (DFA).

    Parameters

    ----------

    x : list or np.array

        One-dimensional time-series.

    Returns

    -------

    dfa : float

        the estimate alpha for the Hurst parameter:

        alpha < 1: stationary process similar to fractional Gaussian noise

        with H = alpha

        alpha > 1: non-stationary process similar to fractional Brownian

        motion with H = alpha - 1

    Notes

    -----

    Detrended fluctuation analysis (DFA) is used to find long-term statistical

    dependencies in time series.

    The idea behind DFA originates from the definition of self-affine

    processes. A process :math:`X` is said to be self-affine if the standard

    deviation of the values within a window of length n changes with the window

    length factor L in a power law:

    .. math:: \\text{std}(X, L * n) = L^H * \\text{std}(X, n)

    where :math:`\\text{std}(X, k)` is the standard deviation of the process

    :math:`X` calculated over windows of size :math:`k`. In this equation,

    :math:`H` is called the Hurst parameter, which behaves indeed very similar

    to the Hurst exponent.

    For more details, please refer to the excellent documentation of the nolds

    Python package by Christopher Scholzel, from which this function is taken:

    https://cschoel.github.io/nolds/nolds.html#detrended-fluctuation-analysis

    Note that the default subseries size is set to

    entropy.utils._log_n(4, 0.1 * len(x), 1.2)). The current implementation

    does not allow to manually specify the subseries size or use overlapping

    windows.

    The code is a faster (Numba) adaptation of the original code by Christopher

    Scholzel.

    References

    ----------

    .. [1] C.-K. Peng, S. V. Buldyrev, S. Havlin, M. Simons,

           H. E. Stanley, and A. L. Goldberger, “Mosaic organization of

           DNA nucleotides,” Physical Review E, vol. 49, no. 2, 1994.

    .. [2] R. Hardstone, S.-S. Poil, G. Schiavone, R. Jansen,

           V. V. Nikulin, H. D. Mansvelder, and K. Linkenkaer-Hansen,

           “Detrended fluctuation analysis: A scale-free view on neuronal

           oscillations,” Frontiers in Physiology, vol. 30, 2012.

    Examples

    --------

        >>> import numpy as np

        >>> from entropy import detrended_fluctuation

        >>> np.random.seed(123)

        >>> x = np.random.rand(100)

        >>> print(detrended_fluctuation(x))

            0.761647725305623

    """

    x = np.asarray(x, dtype=np.float64)

    return _dfa(x)
def extract_features(X_data, segId, serie):

    """Extract some features from a chunk of 150_000, Fill X_data with features"""

    

    xcdm = serie - np.mean(serie) # signal without continuous component

    

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



    b, a = des_bw_filter_bp(low=12500, high=15000) # important (mad)

    xc5 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=15000, high=17500) # important (mad)

    xc6 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_bp(low=17500, high=20000)

    xc7 = sg.lfilter(b, a, xcdm)



    b, a = des_bw_filter_hp(cutoff=20000)

    xc8 = sg.lfilter(b, a, xcdm)

    

    

    sigs = [

        serie,

        pd.Series(xc0),

        pd.Series(xc1),

        pd.Series(xc2),

        pd.Series(xc3),

        pd.Series(xc4),

        pd.Series(xc5),

        pd.Series(xc6),

        pd.Series(xc7),

        pd.Series(xc8)

    ]

    

    for i, sig in enumerate(sigs):

        # TODO features for each filtered signals

        #X_data.loc[segId, "mean_all_%d" % i] = sig.mean()

        #X_data.loc[segId, "std_all_%d" % i] = sig.std() # correlated with mad



        #X_data.loc[segId, "mad_%d" % i] = sig.mad() #important at 0,4,5 and (6,7)-> very important

        #X_data.loc[segId, "kurtosis_all_%d" % i] = sig.kurtosis()

        #X_data.loc[segId, "skew_all_%d" % i] = sig.skew()

        

        #X_data.loc[segId, "fractal_dim_higuchi_%d" % i] = higuchi_fd(sig) # always important

        

        X_data.loc[segId, "hurst_exponent_%d" % i] = nolds.hurst_rs(sig)

        

        

        X_data.loc[segId, "classic_sta_lta_mean_%d" % i] = classic_sta_lta(sig, 100, 5000).mean() # important



        #X_data.loc[segId, "mean_change_rate_%d" % i] = calc_change_rate(sig)

        #X_data.loc[segId, "iqr_%d" % i] = np.subtract(*np.percentile(sig, [75, 25]))

    

    

    X_data.loc[segId, "fractal_dim_higuchi"] = higuchi_fd(serie) # always important

    

    X_data.loc[segId, "mean"] = np.mean(serie)

    X_data.loc[segId, "mad_6"] = pd.Series(xc5).mad() #important at 0,4,5 and (6,7)-> very important

    X_data.loc[segId, "mad_7"] = pd.Series(xc6).mad() #important at 0,4,5 and (6,7)-> very important

    

    X_data.loc[segId, "abs_q05"] = np.quantile(np.abs(serie), 0.05)

    

    X_data.loc[segId,"std_0_to_10"]  = serie[(serie>=0) & (serie<=10)].std()

    X_data.loc[segId,"std_minus_10_to_10"]  = serie[(serie>=-10) & (serie<=10)].std()

    

    for p in [5, 10]:

         X_data.loc[segId, "nb_peaks_%i" % p] = feature_calculators.number_peaks(serie, p)

    

    X_data.loc[segId, "num_crossings"] = feature_calculators.number_crossing_m(serie, 0)

    

    """

    sigFloat = np.float32(serie)

    X_data.loc[segId, "spectral_centroid"] = np.mean(librosa.feature.spectral_centroid(sigFloat))

    X_data.loc[segId, "spectral_bandwidth"] = np.mean(librosa.feature.spectral_bandwidth(sigFloat))

    X_data.loc[segId, "spectral_rolloff"] = np.mean(librosa.feature.spectral_rolloff(sigFloat))

    X_data.loc[segId, "spectral_flatness"] = np.mean(librosa.feature.spectral_flatness(sigFloat))

    X_data.loc[segId, "spectral_contrast"] = np.mean(librosa.feature.spectral_contrast(sigFloat))

    """

    

    x_roll_std_100 = serie.rolling(100).std().dropna().values[::100]

    #X_data.loc[segId, "q05_roll_std_100"] = np.quantile(x_roll_std_100, 0.05) # important

    #X_data.loc[segId, "q15_roll_std_100"] = np.quantile(x_roll_std_100, 0.15) # important 

    X_data.loc[segId, "q30_roll_std_100"] = np.quantile(x_roll_std_100, 0.30) # most important
nrows = 615_000_000

rows_by_chunk = 150_000 # number of rows by chunks

segments = nrows//rows_by_chunk



X_data = pd.DataFrame(index=range(segments), dtype=np.float32)

y_data = pd.DataFrame(index=range(segments), dtype=np.float32)



data_iterator = get_data_iterator(nrows)



for segId in tqdm(range(segments)):

    seg = data_iterator.get_chunk(rows_by_chunk)

    time_to_failure = seg['time_to_failure'].values[-1]

    

    extract_features(X_data, segId, seg['acoustic_data'])

    y_data.loc[segId, "time_to_failure"] = time_to_failure # predict the last value of the chunk

    

y_data = np.array(y_data).flatten()



gc.collect()

X_data.head()
X_data.fillna(value=0, inplace=True)

feature_list = np.array(X_data.columns)

print("Shape of data :", X_data.shape)

feature_list
plt.hist(y_data, bins=17)

plt.xlabel("time_of_failure")

plt.ylabel("occurences")

plt.show()
def plot_corr(df,size=20):

    '''Function plots a graphical correlation matrix for each pair of columns in the dataframe.



    Input:

        df: pandas DataFrame

        size: vertical and horizontal size of the plot'''



    corr = df.corr()

    fig, ax = plt.subplots(figsize=(size, size))

    mat = ax.matshow(corr, cmap="coolwarm")

    fig.colorbar(mat)

    plt.xticks(range(len(corr.columns)), corr.columns, rotation='vertical');

    plt.yticks(range(len(corr.columns)), corr.columns);

    plt.show()
plot_corr(X_data)
np.abs(X_data.corrwith(pd.Series(y_data))).sort_values(ascending=False).head(20)
def plot_feature_vs_ttf(data, ttf, title="No Title"):

    fig, ax1 = plt.subplots(figsize=(16, 8))

    plt.title(title)

    plt.plot(data, color='b')

    ax1.set_ylabel('acoustic_data', color='b')

    plt.legend(['acoustic_data'])

    ax2 = ax1.twinx()

    plt.plot(ttf, color='r')

    ax2.set_ylabel('time_to_failure', color='r')

    plt.legend(['time_to_failure'], loc=(0.875, 0.9))

    plt.grid(False)

    plt.show()
for name, values in X_data.iteritems():

    plot_feature_vs_ttf(values, y_data, name)
def plot_TSNE(data, y_data):

    X_embedded = TSNE(n_components=2).fit_transform(data)



    plt.figure(figsize=(15,10))

    plt.scatter(X_embedded[:,0], X_embedded[:,1], c=y_data, cmap='inferno')

    plt.colorbar()

    plt.title("Data tranformed by TSNE in two dimensions")

    plt.show()

    

    gc.collect()
plot_TSNE(X_data, y_data)
plot_TSNE(StandardScaler().fit_transform(X_data), y_data)
gc.collect()
X_train, X_test, y_train, y_test = train_test_split(

    X_data, y_data,

    test_size=0.2, 

    shuffle=True, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(

    X_train, y_train,

    test_size=0.2,

    shuffle=True, random_state=42

)



print("Training on", X_train.shape[0], "rows")

print("Validating on", X_valid.shape[0], "rows")

print("Testing on", X_test.shape[0], "rows")



f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)

f.suptitle('Histogram of time_to_failure')

ax1.hist(y_train, bins=17)

ax1.set_title("train")

ax2.hist(y_valid, bins=17)

ax2.set_title("test")

ax3.hist(y_test, bins=17)

ax3.set_title("validation")

plt.show()
smt = SMOTE()

X_train, y_train = smt.fit_sample(X_train, np.round(y_train))



plt.hist(y_train, bins=17)

plt.xlabel("time_of_failure")

plt.ylabel("occurences")

plt.show()
scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)

X_valid = scaler.transform(X_valid)

print("Data scaled")
def plotAndScore(y_pred, y_test, name):

    

    score = mean_absolute_error(y_test, y_pred)

    

    plt.figure(figsize=(6, 6))

    plt.scatter(y_test, y_pred)

    plt.xlim(0, 20)

    plt.ylim(0, 20)

    plt.xlabel('actual', fontsize=12)

    plt.ylabel('predicted', fontsize=12)

    plt.title(name + f' -> MAE: {score:0.3f}')

    plt.plot([(0, 0), (20, 20)], [(0, 0), (20, 20)])

    plt.show()

    

    return score

    
dataPool = Pool(X_train, y_train) 

modelCatBoost = CatBoostRegressor(iterations=1000,

                        loss_function='MAE',

                        learning_rate=0.1,

                        depth=6,

                        l2_leaf_reg=3,

                        border_count=128,

                        use_best_model=True,

                        silent=True)



modelCatBoost.fit(dataPool, eval_set=Pool(X_valid, y_valid), plot=True)





y_pred = modelCatBoost.predict(X_test)



score = plotAndScore(y_pred, y_test, "Catboost")
fea_imp = pd.DataFrame({'imp': modelCatBoost.feature_importances_, 'col': feature_list})

fea_imp = fea_imp.sort_values(['imp', 'col'], ascending=[True, False])[-20:]

fea_imp.plot(kind='barh', x='col', y='imp', figsize=(15, 10), legend=None)

plt.title('CatBoost - Feature Importance')

plt.ylabel('Features')

plt.xlabel('Importance')

plt.show()
import shap



shap.initjs()



explainer = shap.TreeExplainer(modelCatBoost)

shap_values = explainer.shap_values(X_train)



# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value, shap_values[0,:], feature_list)
shap.summary_plot(shap_values, X_train, feature_names=feature_list)
shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=feature_list)
from catboost import cv

params = {}

params["iterations"] = 500

params["loss_function"] = 'MAE'

params["learning_rate"] = 0.2

params["depth"] = 6

params["verbose"] = 0



cv_data = cv(pool=Pool(StandardScaler().fit_transform(X_data), y_data), 

   params=params,

   fold_count=5, 

   inverted=False,

   shuffle=True, 

   stratified=False,

   as_pandas=True,

   plot=True,

   verbose=False)



min_test_mean = cv_data["test-MAE-mean"].min()

print("best MAE :",min_test_mean)
models = {

    "CatBoostRegressor" : CatBoostRegressor(iterations=300, loss_function='MAE', learning_rate=0.2, depth=6, verbose=0),

    "NuSVR" : NuSVR(kernel="rbf", C=1),

    "SVR" : SVR(kernel="rbf", C=1),

    "DecisionTreeRegressor" : DecisionTreeRegressor(),

    "RandomForestRegressor" : RandomForestRegressor(max_depth=6, n_estimators=200),

    "GradientBoostingRegressor" : GradientBoostingRegressor(),

    "KNeighborsRegressor(n_neighbors=50)" : KNeighborsRegressor(n_neighbors=50),

    'LinearRegression': LinearRegression(),

    'Ridge_1': Ridge(alpha=0.001),

    'Lasso_1': Lasso(alpha=0.001),

    'ElasticNet_1': ElasticNet(alpha=0.001, l1_ratio=0),

    'ElasticNet_2': ElasticNet(alpha=0.001, l1_ratio=0.25),

    'ElasticNet_3': ElasticNet(alpha=0.001, l1_ratio=0.33),

    'ElasticNet_4': ElasticNet(alpha=0.001, l1_ratio=0.5),

    'ElasticNet_5': ElasticNet(alpha=0.001, l1_ratio=0.66),

    'ElasticNet_6': ElasticNet(alpha=0.001, l1_ratio=0.75),

    'ElasticNet_7': ElasticNet(alpha=0.001, l1_ratio=1),

}
best_model_name = ""

best_score = 100



for name, model in tqdm(models.items()):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    new_score = plotAndScore(y_pred, y_test, name)

    if new_score < best_score:

        best_score = new_score

        best_model_name = name
best_model_name
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_data_submission = pd.DataFrame(dtype=np.float64, index=submission.index)



plot_data_submission = pd.Series()



for seg_id in tqdm(X_data_submission.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    

    plot_data_submission.append(seg[::50])

    

    x = seg['acoustic_data']

    extract_features(X_data_submission, seg_id, x)

X_data_submission = scaler.transform(X_data_submission)

submission['time_to_failure'] = [x*1.05 for x in modelCatBoost.predict(X_data_submission)]
# TODO

acoustic_data_small = []



for seg_id in tqdm(submission.index):

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    acoustic_data_small.append(seg['acoustic_data'][::50])



time_to_failure_small = submission["time_to_failure"].values



fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title("Trends of acoustic_data and time_to_failure. 2% of data (sampled)")

plt.plot(acoustic_data_small, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(time_to_failure_small, color='r', alpha=0.5)

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)

plt.show()
del acoustic_data_small

del time_to_failure_small

gc.collect()
submission.to_csv('submission.csv')

print("Submission data saved to csv.")