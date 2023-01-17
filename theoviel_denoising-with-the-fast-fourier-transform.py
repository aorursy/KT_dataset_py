import numpy as np

import pandas as pd

import seaborn as sns

from numpy.fft import *

import pyarrow.parquet as pq

import matplotlib.pyplot as plt
df_train = pd.read_csv('../input/liverpool-ion-switching/train.csv')

df_train.head()
n_times = 1000

time = df_train['time'][:n_times].values

signal = df_train['signal'][:n_times].values
plt.figure(figsize=(15, 10))

plt.plot(time, signal)

plt.title('Signal', size=15)

plt.show()
def filter_signal(signal, threshold=1e8):

    fourier = rfft(signal)

    frequencies = rfftfreq(signal.size, d=1e-5)

    fourier[frequencies > threshold] = 0

    return irfft(fourier)
for threshold in [1e3, 5e3, 1e4, 5e4]:

    filtered = filter_signal(signal, threshold=threshold)



    plt.figure(figsize=(15, 10))

    plt.plot(signal, label='Raw')

    plt.plot(filtered, label='Filtered')

    plt.legend()

    plt.title(f"FFT Denoising with threshold = {threshold :.0e}", size=15)

    plt.show()