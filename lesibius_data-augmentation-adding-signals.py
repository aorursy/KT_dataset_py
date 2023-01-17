import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

labels = pd.read_csv('../input/train.csv')
labels[labels['label'] == 'Trumpet'].head()
labels[labels['label'] == 'Cello'].head()
trumpet = '034e4ffa'
cello = '00353774'
import IPython.display as ipd  # To play sound in the notebook
t_fname = '../input/audio_train/' + trumpet + '.wav'
ipd.Audio(t_fname)

import IPython.display as ipd  # To play sound in the notebook
c_fname = '../input/audio_train/' + cello + '.wav'
ipd.Audio(c_fname)
from scipy.io import wavfile
rate, t_signal = wavfile.read(t_fname)
rate, c_signal = wavfile.read(c_fname)

min_len = min(len(t_signal),len(c_signal))

t_signal = np.array([(e/2**16.0)*2 for e in t_signal]) #16 bits tracks, normalization
c_signal = np.array([(e/2**16.0)*2 for e in c_signal])

t_signal = t_signal[:min_len]
c_signal = c_signal[:min_len]

new_sig = t_signal + c_signal

new_sig_16b = np.array([int((v*2**16.0)/2) for v in new_sig])
plt.plot(new_sig_16b)

ipd.Audio(new_sig_16b,rate=44100)


labels[labels['label'] == 'Acoustic_guitar'].head()
g1 = '0356dec7'
g1_fname = '../input/audio_train/' + g1 + '.wav'
ipd.Audio(g1_fname)
g2 = '0969b5c5'
g2_fname = '../input/audio_train/' + g2 + '.wav'
ipd.Audio(g2_fname)
rate, g1_signal = wavfile.read(g1_fname)
rate, g2_signal = wavfile.read(g2_fname)

min_len = min(len(g1_signal),len(g2_signal))

g1_signal = np.array([(e/2**16.0)*2 for e in g1_signal]) #16 bits tracks, normalization
g2_signal = np.array([(e/2**16.0)*2 for e in g2_signal])

g1_signal = g1_signal[:min_len]
g2_signal = g2_signal[:min_len]

new_sig = g1_signal + g2_signal

new_sig_16b = np.array([int((v*2**16.0)/2) for v in new_sig])
g1_sig_16b = np.array([int((v*2**16.0)/2) for v in g1_signal])
g2_sig_16b = np.array([int((v*2**16.0)/2) for v in g2_signal])
plt.plot(new_sig_16b)

ipd.Audio(new_sig_16b,rate=44100)


from scipy import signal
#data
freqs, times, specs = signal.spectrogram(new_sig,
                                         fs=44100,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')
freqs, times, specs1 = signal.spectrogram(g1_signal,
                                         fs=44100,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')
freqs, times, specs2 = signal.spectrogram(g2_signal,
                                         fs=44100,
                                         window="boxcar",
                                        nperseg=13230,
                                        noverlap=0,
                                        detrend=False,
                                        mode = 'complex')

max_freq = 1000
plt.plot(freqs[:max_freq],np.absolute(specs[:max_freq,1]),'r')
plt.plot(freqs[:max_freq],np.absolute(specs1[:max_freq,1]),'bo')
plt.plot(freqs[:max_freq],np.absolute(specs2[:max_freq,1]),'gx')