import numpy as np

import pandas as pd

from scipy import signal

from scipy.fft import fftshift

from tqdm import tqdm_notebook as tqdm



# visualize

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

from matplotlib import pyplot

from matplotlib.ticker import ScalarFormatter

sns.set_context("talk")

style.use('fivethirtyeight')
# load data

df_train = pd.read_csv("../input/data-without-drift/train_clean.csv")

train_time   = df_train["time"].values.reshape(-1,500000)

train_signal = df_train["signal"].values.reshape(-1,500000)

train_opench = df_train["open_channels"].values.reshape(-1,500000)

# df_test = pd.read_csv("../input/data-without-drift/test_clean.csv")

# test_time   = df_test["time"].values.reshape(-1,500000)

# test_signal = df_test["signal"].values.reshape(-1,500000)
# # sample data for quick test

# train_time = train_time[:, ::100]

# train_signal = train_signal[:, ::100]

# train_opench = train_opench[:, ::100]
train_signal.shape
def spectrogram_plot(train_signal, train_opench, i):

    fig, ax = plt.subplots(3, 1, figsize=(8, 8), 

                           gridspec_kw={"height_ratios": [1, 3, 1]})

    ax = ax.flatten()

    

    # open channels

    ax[0].plot(np.arange(500_000), train_opench[i], lw=0.05, color='r')

    ax[0].set_title(f"batch {i}")

    ax[0].set_ylabel("open channels")

    ax[0].set_xlim([0, 500_000])

    

    # spectrogram

    fs = 10_000 # sampling rate is 10kHz

    f, t, Sxx = signal.spectrogram(train_signal[i], fs)

    ax[1].pcolormesh(t, f, -np.log(Sxx), cmap="plasma")

    ax[1].set_ylabel('Frequency [Hz]')

    ax[1].set_ylim([0, 500])

    ax[1].set_xlabel('Time [sec]')

    plt.tight_layout()

    

    # Power histogram (collapsed across time)

    ax[2].plot(f, np.mean(Sxx, axis=1), color="g")

    ax[2].set_xlabel("Frequency [Hz]")

    ax[2].set_xlim([0, 500])

    ax[2].set_ylabel("Power")

    

spectrogram_plot(train_signal, train_opench, 0)
spectrogram_plot(train_signal, train_opench, 1)
spectrogram_plot(train_signal, train_opench, 2)
spectrogram_plot(train_signal, train_opench, 3)
spectrogram_plot(train_signal, train_opench, 4)
spectrogram_plot(train_signal, train_opench, 5)
spectrogram_plot(train_signal, train_opench, 6)
spectrogram_plot(train_signal, train_opench, 7)
spectrogram_plot(train_signal, train_opench, 8)
spectrogram_plot(train_signal, train_opench, 9)