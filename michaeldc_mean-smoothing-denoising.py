import numpy as np

import matplotlib.pyplot as plt

import scipy.io as sio

import scipy.signal

from scipy import *

import copy
srate = 1000 #Hz sample rate -> Number of samples taken by sec

time  = np.arange(0,3,1/srate) 

n     = len(time) 

p     = 15 # poles for random interpolation



# Noise level, measured in standard deviations

noiseamp = 5



# Amp modulator and noise level

ampl   = np.interp(np.linspace(0, p, n), np.arange(0,p), np.random.rand(p)*30)

noise  = noiseamp * np.random.randn(n)

signal = ampl + noise



# Init filter vector with zeros

filtsig = np.zeros(n)
plt.plot(time, signal, label='noisy')

plt.legend()

plt.xlabel('Time (sec.)')

plt.ylabel('Amplitude')

plt.title('Noise Signal')

plt.show()
k = 30 # filter window is actually k*2+1

for i in range (k+1, n-k-1):

    filtsig[i] = np.mean(signal[i-k:i+k])



windowsize = 1000*(k*2+1) / srate

plt.plot(time, signal, label='original')

plt.plot(time, filtsig, label='filtered')



plt.legend()

plt.xlabel('Time (sec.)')

plt.ylabel('Amplitude')

plt.title('Filtered with a k=%d-ms filter Window' %windowsize)

plt.show()