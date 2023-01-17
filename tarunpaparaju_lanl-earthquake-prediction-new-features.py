import os

import gc

import numpy as np

import pandas as pd

from tqdm import tqdm

tqdm.pandas()



from numba import jit

from math import log, floor

from sklearn.neighbors import KDTree

from scipy.signal import periodogram, welch



from keras.layers import *

from keras.models import *

from tqdm import tqdm

from sklearn.model_selection import train_test_split 

from keras import backend as K

from keras import optimizers

from sklearn.model_selection import GridSearchCV, KFold

from keras.callbacks import *

from keras import activations

from keras import regularizers

from keras import initializers

from keras import constraints

from keras.engine import Layer

from keras.engine import InputSpec

from keras.objectives import categorical_crossentropy

from keras.objectives import sparse_categorical_crossentropy

from keras.utils import plot_model

from keras.utils.vis_utils import model_to_dot



import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import SVG



import warnings

warnings.filterwarnings('ignore')
SIGNAL_LEN = 150000

MIN_NUM = -27

MAX_NUM = 28
seismic_signals = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
acoustic_data = seismic_signals.acoustic_data

time_to_failure = seismic_signals.time_to_failure

data_len = len(seismic_signals)

del seismic_signals

gc.collect()
signals = []

targets = []



for i in range(data_len//SIGNAL_LEN):

    min_lim = SIGNAL_LEN * i

    max_lim = min([SIGNAL_LEN * (i + 1), data_len])

    

    signals.append(list(acoustic_data[min_lim : max_lim]))

    targets.append(time_to_failure[max_lim])

    

del acoustic_data

del time_to_failure

gc.collect()

    

signals = np.array(signals)

targets = np.array(targets)
def min_max_transfer(ts, min_value, max_value, range_needed=(-1,1)):

    ts_std = (ts - min_value) / (max_value - min_value)



    if range_needed[0] < 0:    

        return ts_std * (range_needed[1] + abs(range_needed[0])) + range_needed[0]

    else:

        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]
def transform_ts(ts, n_dim=160, min_max=(-1,1)):

    ts_std = min_max_transfer(ts, min_value=MIN_NUM, max_value=MAX_NUM)

    bucket_size = int(SIGNAL_LEN / n_dim)

    new_ts = []

    for i in range(0, SIGNAL_LEN, bucket_size):

        ts_range = ts_std[i:i + bucket_size]

        mean = ts_range.mean()

        std = ts_range.std()

        std_top = mean + std

        std_bot = mean - std

        percentil_calc = ts_range.quantile([0, 0.01, 0.25, 0.50, 0.75, 0.99, 1])

        max_range = ts_range.quantile(1) - ts_range.quantile(0)

        relative_percentile = percentil_calc - mean

        new_ts.append(np.concatenate([np.asarray([mean, std, std_top, std_bot, max_range]), percentil_calc, relative_percentile]))

    return np.asarray(new_ts)
def prepare_data(start, end):

    train = pd.DataFrame(np.transpose(signals[int(start):int(end)]))

    X = []

    for id_measurement in tqdm(train.index[int(start):int(end)]):

        X_signal = transform_ts(train[id_measurement])

        X.append(X_signal)

    X = np.asarray(X)

    return X
X = []



def load_all():

    total_size = len(signals)

    for start, end in [(0, int(total_size))]:

        X_temp = prepare_data(start, end)

        X.append(X_temp)

        

load_all()

X = np.concatenate(X)
X.shape
shape = X.shape

new_signals = X.reshape((shape[0], shape[1]*shape[2]))



sparse_signals = []

for i in range(3):

    sparse_signal = []

    for j in range(len(new_signals[i])):

        if j % 3 == 0:

            sparse_signal.append(new_signals[i][j])

    sparse_signals.append(sparse_signal)



plt.plot(sparse_signals[0], 'mediumseagreen')

plt.show()

plt.plot(sparse_signals[1], 'seagreen')

plt.show()

plt.plot(sparse_signals[2], 'green')

plt.show()
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
spectral_entropies = np.array([spectral_entropy(new_signal, sf=100, method='fft') for new_signal in new_signals])
plot = sns.jointplot(x=spectral_entropies, y=targets, kind='kde', color='blueviolet')

plot.set_axis_labels('spectral_entropy', 'time_to_failure', fontsize=16)

plt.show()
plot = sns.jointplot(x=spectral_entropies, y=targets, kind='hex', color='blueviolet')

plot.set_axis_labels('spectral_entropy', 'time_to_failure', fontsize=16)

plt.show()
plot = sns.jointplot(x=spectral_entropies, y=targets, kind='reg', color='blueviolet')

plot.set_axis_labels('spectral_entropy', 'time_to_failure', fontsize=16)

plt.show()
@jit('f8(f8[:], i4, f8)', nopython=True)

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



def sample_entropy(x, order=2, metric='chebyshev'):

    """Sample Entropy.

    Parameters

    ----------

    x : list or np.array

        One-dimensional time series of shape (n_times)

    order : int (default: 2)

        Embedding dimension.

    metric : str (default: chebyshev)

        Name of the metric function used with KDTree. The list of available

        metric functions is given by: `KDTree.valid_metrics`.

    Returns

    -------

    se : float

        Sample Entropy.

    Notes

    -----

    Sample entropy is a modification of approximate entropy, used for assessing

    the complexity of physiological time-series signals. It has two advantages

    over approximate entropy: data length independence and a relatively

    trouble-free implementation. Large values indicate high complexity whereas

    smaller values characterize more self-similar and regular signals.

    Sample entropy of a signal :math:`x` is defined as:

    .. math:: H(x, m, r) = -log\\frac{C(m + 1, r)}{C(m, r)}

    where :math:`m` is the embedding dimension (= order), :math:`r` is

    the radius of the neighbourhood (default = :math:`0.2 * \\text{std}(x)`),

    :math:`C(m + 1, r)` is the number of embedded vectors of length

    :math:`m + 1` having a Chebyshev distance inferior to :math:`r` and

    :math:`C(m, r)` is the number of embedded vectors of length

    :math:`m` having a Chebyshev distance inferior to :math:`r`.

    Note that if metric == 'chebyshev' and x.size < 5000 points, then the

    sample entropy is computed using a fast custom Numba script. For other

    metric types or longer time-series, the sample entropy is computed using

    a code from the mne-features package by Jean-Baptiste Schiratti

    and Alexandre Gramfort (requires sklearn).

    References

    ----------

    .. [1] Richman, J. S. et al. (2000). Physiological time-series analysis

           using approximate entropy and sample entropy. American Journal of

           Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.

    Examples

    --------

    1. Sample entropy with order 2.

        >>> from entropy import sample_entropy

        >>> import numpy as np

        >>> np.random.seed(1234567)

        >>> x = np.random.rand(3000)

        >>> print(sample_entropy(x, order=2))

            2.192

    2. Sample entropy with order 3 using the Euclidean distance.

        >>> from entropy import sample_entropy

        >>> import numpy as np

        >>> np.random.seed(1234567)

        >>> x = np.random.rand(3000)

        >>> print(sample_entropy(x, order=3, metric='euclidean'))

            2.725

    """

    x = np.asarray(x, dtype=np.float64)

    if metric == 'chebyshev' and x.size < 5000:

        return _numba_sampen(x, mm=order, r=0.2)

    else:

        phi = _app_samp_entropy(x, order=order, metric=metric,

                                approximate=False)

        return -np.log(np.divide(phi[1], phi[0]))
sample_entropies = np.array([sample_entropy(new_signal) for new_signal in new_signals])
plot = sns.jointplot(x=sample_entropies, y=targets, kind='kde', color='mediumvioletred')

plot.set_axis_labels('sample_entropy', 'time_to_failure', fontsize=16)

plt.show()
plot = sns.jointplot(x=sample_entropies, y=targets, kind='hex', color='mediumvioletred')

plot.set_axis_labels('sample_entropy', 'time_to_failure', fontsize=16)

plt.show()
plot = sns.jointplot(x=sample_entropies, y=targets, kind='reg', color='mediumvioletred')

plot.set_axis_labels('sample_entropy', 'time_to_failure', fontsize=16)

plt.show()
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
detrended_fluctuations = np.array([detrended_fluctuation(new_signal) for new_signal in new_signals])
plot = sns.jointplot(x=detrended_fluctuations, y=targets, kind='kde', color='mediumblue')

plot.set_axis_labels('detrended_fluctuation', 'time_to_failure', fontsize=16)

plt.show()
plot = sns.jointplot(x=detrended_fluctuations, y=targets, kind='hex', color='mediumblue')

plot.set_axis_labels('detrended_fluctuation', 'time_to_failure', fontsize=16)

plt.show()
plot = sns.jointplot(x=detrended_fluctuations, y=targets, kind='reg', color='mediumblue')

plot.set_axis_labels('detrended_fluctuation', 'time_to_failure', fontsize=16)

plt.show()