import numpy as np

import pandas as pd

from pandas import DataFrame

from scipy.io import loadmat



import matplotlib.pyplot as plt



def mat_to_dataframe(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])



X0 = mat_to_dataframe('../input/train_1/1_1_1.mat')

plt.plot(X0.index, X0[3])

plt.show()
X = X0.iloc[:, 1]

Fs = 400.0

time_stamp = 1.0/Fs

t = np.arange(0, len(X)*time_stamp, time_stamp)

sp = np.fft.fft(X)

freq = np.fft.fftfreq(len(X), d=time_stamp)

plt.plot(freq, sp.real)

window = np.hamming(128)

plt.plot(window)