import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile
import os
from scipy import signal


def plot_signal(x):
    x = x/(.8*max(x))
    # Show one recording
    plt.subplot(211)
    #plt.figure(figsize = (10,8))
    plt.plot(np.linspace(0,len(x)/fs,len(x)),x)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('time (s)')
    plt.title('vibration recording')
    plt.grid(True)
    
    plt.subplot(212)
    cmap = plt.get_cmap('viridis')
    vmin = 20*np.log10(np.max(x)) - 80  # hide anything below -40 dBc
    cmap.set_under(color='k', alpha=None)

    #plt.figure(figsize = (10,8))
    Pxx, freqs, bins, im = plt.specgram(x, NFFT=256, Fs=fs, noverlap=256-256/6, cmap=cmap, vmin=vmin)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel('time (s)')
    plt.ylabel('frequency [Hz]')
    plt.show()
    return

path = '/kaggle/input/lab/lab/infested/infested_1.wav' 
fs, x = scipy.io.wavfile.read(path)
plot_signal(x)
path = '/kaggle/input/field/field/train/infested/folder_1/F_20200218135459_84_T24.4.wav' 
fs, x = scipy.io.wavfile.read(path)
plot_signal(x)
