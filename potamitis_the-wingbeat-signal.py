from __future__ import division

import numpy as np

from matplotlib import pyplot as plt

import scipy.io.wavfile

import os

from scipy import signal



fs, x = scipy.io.wavfile.read('../input/Wingbeats/Wingbeats/Ae. aegypti/D_16_12_12_19_46_13/F161212_194613_156_G_050.wav')

x = x/max(x)

X=10*np.log10(signal.welch(x, fs=fs, window='hanning', nperseg=256, noverlap=128+64)[1])



# Show one recording

plt.figure(figsize = (10,8))

plt.plot(np.linspace(0,len(x)/fs,len(x)),x)

plt.autoscale(enable=True, axis='x', tight=True)

plt.xlabel('time (s)')

plt.title('Wingbeat recording')

plt.show()
plt.figure(figsize = (10,8))

plt.plot(np.linspace(0,fs/2,129),X)

plt.autoscale(enable=True, axis='x', tight=True)

plt.xlabel('frequency [Hz]')

plt.ylabel('PSD [dB]')

plt.grid(True)

plt.title('Welch Power Spectral Density')

plt.show()
plt.figure(figsize = (10,8))

Pxx, freqs, bins, im = plt.specgram(x, NFFT=256, Fs=fs, noverlap=256-256/6)

plt.autoscale(enable=True, axis='x', tight=True)

plt.xlabel('time (s)')

plt.ylabel('frequency [Hz]')

plt.show()