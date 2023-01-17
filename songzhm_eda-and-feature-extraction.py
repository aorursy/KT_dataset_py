# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.io import loadmat

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



import matplotlib

import matplotlib.pyplot as plt


# Any results you write to the current directory are saved as output

labels = pd.read_csv('../input/train_and_test_data_labels_safe.csv')



# find out how many safe data clips are available for each patients

for i in range(1,4):

    safe_labels =labels.loc[(labels.safe == 1) & (labels.image.str.contains('{id}_[0-9]+_'.format(id = i)))]

    print(i,':',len(safe_labels))
safe_labels['class']
def mat_to_data(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return ndata

def ieegMatToPandasDF(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    print(ndata)

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])   





# choose 2nd patients to start

safe_labels2 =labels.loc[(labels.safe == 1) & (labels.image.str.contains('{id}_[0-9]+_'.format(id = 2)))]



#path = '../input/train_2/'

path = '../input/test_1_new/'



id = 3634

label = safe_labels2['class'][id]

mat_path = path + safe_labels2['image'][id]

# mat = loadmat(mat_path)

# names = mat['dataStruct'].dtype.names

# ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

# data = pd.DataFrame(ndata['data'],columns = [str(x+1) for x in range(16)])

data=ieegMatToPandasDF(mat_path)

print(label)

print(data.shape)
%matplotlib inline

matplotlib.rcParams['figure.figsize'] = (30, 30)

n=16

for i in range(0, n):

#     print i

    plt.subplot(n, 1, i + 1)

    plt.plot(data[i +1])
# the following code is borrowed form the source code of pyeeg python library:

# https://github.com/forrestbao/pyeeg



import numpy



def bin_power(X, Band, Fs):

    """Compute power in each frequency bin specified by Band from FFT result of

    X. By default, X is a real signal.



    Note

    -----

    A real signal can be synthesized, thus not real.



    Parameters

    -----------



    Band

        list



        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.

        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.

        You can also use range() function of Python to generate equal bins and

        pass the generated list to this function.



        Each element of Band is a physical frequency and shall not exceed the

        Nyquist frequency, i.e., half of sampling frequency.



     X

        list



        a 1-D real time series.



    Fs

        integer



        the sampling rate in physical frequency



    Returns

    -------



    Power

        list



        spectral power in each frequency bin.



    Power_ratio

        list



        spectral power in each frequency bin normalized by total power in ALL

        frequency bins.



    """



    C = numpy.fft.fft(X)

    C = abs(C)

    Power = numpy.zeros(len(Band) - 1)

    for Freq_Index in range(0, len(Band) - 1):

        Freq = float(Band[Freq_Index])

        Next_Freq = float(Band[Freq_Index + 1])

        Power[Freq_Index] = sum(

            C[numpy.floor(

                Freq / Fs * len(X)

            ): numpy.floor(Next_Freq / Fs * len(X))]

        )

    Power_Ratio = Power / sum(Power)

    return Power, Power_Ratio



x = list(data.iloc[:,0])

D = numpy.diff(x)

D = D.tolist()

power,power_ratio = bin_power(x,[0.1, 4, 8, 14, 30, 45, 70,180],8)



print(power)

print(power_ratio)




def pfd(X, D=None):

    """Compute Petrosian Fractal Dimension of a time series from either two

    cases below:

        1. X, the time series of type list (default)

        2. D, the first order differential sequence of X (if D is provided,

           recommended to speed up)



    In case 1, D is computed using Numpy's difference function.



    To speed up, it is recommended to compute D before calling this function

    because D may also be used by other functions whereas computing it here

    again will slow down.

    """

    if D is None:

        D = numpy.diff(X)

        D = D.tolist()

    N_delta = 0  # number of sign changes in derivative of the signal

    for i in range(1, len(D)):

        if D[i] * D[i - 1] < 0:

            N_delta += 1

    n = len(X)

    return numpy.log10(n) / (

        numpy.log10(n) + numpy.log10(n / n + 0.4 * N_delta)

    )



pfd_res =pfd(x,D)

print(pfd_res)
def hfd(X, Kmax=8):

    """ Compute Hjorth Fractal Dimension of a time series X, kmax

     is an HFD parameter

    """

    L = []

    x = []

    N = len(X)

    for k in range(1, Kmax):

        Lk = []

        for m in range(0, k):

            Lmk = 0

            for i in range(1, int(numpy.floor((N - m) / k))):

                Lmk += abs(X[m + i * k] - X[m + i * k - k])

            Lmk = Lmk * (N - 1) / numpy.floor((N - m) / float(k)) / k

            Lk.append(Lmk)

        L.append(numpy.log(numpy.mean(Lk)))

        x.append([numpy.log(float(1) / k), 1])



    (p, r1, r2, s) = numpy.linalg.lstsq(x, L)

    return p[0]



hfd_res = hfd(x)

print(hfd_res)
def hjorth(X, D=None):

    """ Compute Hjorth mobility and complexity of a time series from either two

    cases below:

        1. X, the time series of type list (default)

        2. D, a first order differential sequence of X (if D is provided,

           recommended to speed up)



    In case 1, D is computed using Numpy's Difference function.



    Notes

    -----

    To speed up, it is recommended to compute D before calling this function

    because D may also be used by other functions whereas computing it here

    again will slow down.



    Parameters

    ----------



    X

        list



        a time series



    D

        list



        first order differential sequence of a time series



    Returns

    -------



    As indicated in return line



    Hjorth mobility and complexity



    """



    if D is None:

        D = numpy.diff(X)

        D = D.tolist()



    D.insert(0, X[0])  # pad the first difference

    D = numpy.array(D)



    n = len(X)



    M2 = float(sum(D ** 2)) / n

    TP = sum(numpy.array(X) ** 2)

    M4 = 0

    for i in range(1, len(D)):

        M4 += (D[i] - D[i - 1]) ** 2

    M4 = M4 / n



    return numpy.sqrt(M2 / TP), numpy.sqrt(

        float(M4) * TP / M2 / M2

    )  # Hjorth Mobility and Complexity



hjorth_res = hjorth(x,D)

print(hjorth_res)
def spectral_entropy(X, Band, Fs, Power_Ratio=None):

    """Compute spectral entropy of a time series from either two cases below:

    1. X, the time series (default)

    2. Power_Ratio, a list of normalized signal power in a set of frequency

    bins defined in Band (if Power_Ratio is provided, recommended to speed up)



    In case 1, Power_Ratio is computed by bin_power() function.



    Notes

    -----

    To speed up, it is recommended to compute Power_Ratio before calling this

    function because it may also be used by other functions whereas computing

    it here again will slow down.



    Parameters

    ----------



    Band

        list



        boundary frequencies (in Hz) of bins. They can be unequal bins, e.g.

        [0.5,4,7,12,30] which are delta, theta, alpha and beta respectively.

        You can also use range() function of Python to generate equal bins and

        pass the generated list to this function.



        Each element of Band is a physical frequency and shall not exceed the

        Nyquist frequency, i.e., half of sampling frequency.



     X

        list



        a 1-D real time series.



    Fs

        integer



        the sampling rate in physical frequency



    Returns

    -------



    As indicated in return line



    See Also

    --------

    bin_power: pyeeg function that computes spectral power in frequency bins



    """



    if Power_Ratio is None:

        Power, Power_Ratio = bin_power(X, Band, Fs)



    Spectral_Entropy = 0

    for i in range(0, len(Power_Ratio) - 1):

        Spectral_Entropy += Power_Ratio[i] * numpy.log(Power_Ratio[i])

    Spectral_Entropy /= numpy.log(

        len(Power_Ratio)

    )  # to save time, minus one is omitted

    return -1 * Spectral_Entropy