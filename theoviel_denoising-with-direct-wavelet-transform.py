import pywt

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



warnings.filterwarnings("ignore")
df_train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

df_train.head()
n_times = 1000

time = df_train['time'][:n_times].values

signal = df_train['signal'][:n_times].values
plt.figure(figsize=(15, 10))

plt.plot(time, signal)

plt.title('Signal', size=15)

plt.show()
def madev(d, axis=None):

    """ Mean absolute deviation of a signal """

    return np.mean(np.absolute(d - np.mean(d, axis)), axis)
def wavelet_denoising(x, wavelet='db4', level=1):

    coeff = pywt.wavedec(x, wavelet, mode="per")

    sigma = (1/0.6745) * madev(coeff[-level])

    uthresh = sigma * np.sqrt(2 * np.log(len(x)))

    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    return pywt.waverec(coeff, wavelet, mode='per')
for wav in pywt.wavelist():

    try:

        filtered = wavelet_denoising(signal, wavelet=wav, level=1)

    except:

        pass

    

    plt.figure(figsize=(10, 6))

    plt.plot(signal, label='Raw')

    plt.plot(filtered, label='Filtered')

    plt.legend()

    plt.title(f"DWT Denoising with {wav} Wavelet", size=15)

    plt.show()