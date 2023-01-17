import numpy as np

import matplotlib.pyplot as plt

import scipy.io as sio

import scipy.signal

from scipy import *

import copy
def gaussian(t, fwhm):

    return np.exp(-(4*np.log(2)*t**2)/fwhm**2)
srate = 1000 #Hz

time  = np.arange(0,3,1/srate)

n     = len(time)

p     = 15 # poles for random interpolation



#noise level, measured in standard deviations

noiseamp = 5



# amplitude modular and noise level

ampl   = np.interp(linspace(1,p,n), np.arange(0,p), np.random.rand(p)*30)

noise  = noiseamp * np.random.randn(n)

signal = ampl + noise
fwhm    = 25 # in ms



# normalized time vector in ms

k       = 100

gtime   = 1000*np.arange(-k, k)/srate



# Gaussian window

gauswin = gaussian(gtime, fwhm)



# Compute empirical FWHM

pstPeakHalf = k + np.argmin( (gauswin[k:] -.5)**2 )

prePeakHalf = np.argmin( (gauswin -.5)**2 )



empFWHM = gtime[pstPeakHalf] - gtime[prePeakHalf]



#show the Gaussian

plt.plot(gtime, gauswin, 'ko-')

plt.plot([gtime[prePeakHalf],gtime[pstPeakHalf]],[gauswin[prePeakHalf],gauswin[pstPeakHalf]],'m')

# Normalize Gaussian to unit energy

gauswin = gauswin/np.sum(gauswin)

#title

plt.xlabel('Time (ms)')

plt.ylabel('Gain')



plt.show()
# initialize filtered signal vector

filtsigG = copy.deepcopy(signal)



# implement the running mean filter

for i in range(k+1, n-k-1):

    filtsigG[i] = np.sum(signal[i-k:i+k] * gauswin)

    

plt.plot(time, signal, 'r', label='Original')

plt.plot(time, filtsigG, 'k', label='Gaussian-filtered')



plt.xlabel('Time (s)')

plt.ylabel('amp. (a.u)')

plt.legend()

plt.title('Gaussian smothing filter')



plt.show()
filtsigMean = copy.deepcopy(signal)



mk = 20

for i in range(mk+1, n-mk-1):

    filtsigMean[i] = mean(signal[i-mk:i+mk])

plt.plot(time,signal,'r',label='Original')    

plt.plot(time, filtsigMean, 'b', label='Running mean')

plt.legend()

plt.show()