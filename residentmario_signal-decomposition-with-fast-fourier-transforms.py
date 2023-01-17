%%time

from numpy.fft import rfft, irfft, rfftfreq
from scipy import fftpack
import pandas as pd

train_meta_df = pd.read_csv("../input/metadata_train.csv").set_index('signal_id')
train = pd.read_parquet("../input/train.parquet")
y = train_meta_df.target
import matplotlib.pyplot as plt

def low_pass(s, threshold=1e4):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2 / s.size)
    fourier[frequencies > threshold] = 0
    return irfft(fourier)

lf_signal_1 = low_pass(train.iloc[:, 0])
plt.plot(train.iloc[:, 0], color='lightgray')
plt.plot(lf_signal_1, color='black')
def high_pass(s, threshold=1e7):
    fourier = rfft(s)
    frequencies = rfftfreq(s.size, d=2e-2/s.size)
    fourier[frequencies < threshold] = 0
    return irfft(fourier)

hf_signal_1 = high_pass(train.iloc[:,0], threshold=1e4)

plt.plot(hf_signal_1)
plt.plot(train.iloc[: 0], color='black')
plt.plot(lf_signal_1 + hf_signal_1, color='lightgray')
plt.plot(rfftfreq(train.iloc[:, 0].size, d=2e-2 / train.iloc[:, 0].size))
import numpy as np

def decompose_into_n_signals(srs, n):
    fourier = rfft(srs)
    frequencies = rfftfreq(srs.size, d=2e-2/srs.size)
    out = []
    for vals in np.array_split(frequencies, n):
        ft_threshed = fourier.copy()
        ft_threshed[(vals.min() > frequencies)] = 0
        ft_threshed[(vals.max() < frequencies)] = 0        
        out.append(irfft(ft_threshed))
    return out

def plot_n_signals(sigs):
    fig, axarr = plt.subplots(len(sigs), figsize=(12, 12))
    for i, sig in enumerate(sigs):
        plt.sca(axarr[i])
        plt.plot(sig)
    plt.gcf().suptitle(f"Decomposition of signal into {len(sigs)} frequency bands", fontsize=24)
plot_n_signals(decompose_into_n_signals(train.iloc[:,0], 5))