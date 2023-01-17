import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import sys

from scipy.signal import hilbert

from PyEMD import EMD

pd.options.display.precision = 10

from os import listdir

print(listdir("../input"))

def instant_phase(imfs):

    """Extract analytical signal through Hilbert Transform."""

    analytic_signal = hilbert(imfs)  # Apply Hilbert transform to each row

    # Compute angle between img and real

    phase = np.unwrap(np.angle(analytic_signal))

    return phase
signal = pd.read_csv('../input/Signal01.csv')

print(signal.head())

S = signal.signal.values[::10]

t = signal.quaketime.values[::10]

print('S shape: ', S.shape)

print('t shape: ', t.shape)
dt = t[0] - t[1]

print(dt)
# Compute IMFs with EMD

config = {'spline_kind':'linear', 'MAX_ITERATION':100}

emd = EMD(**config)

imfs = emd(S, max_imf=10)

print('imfs = ' + f'{imfs.shape[0]:4d}')
# Extract instantaneous phases and frequencies using Hilbert transform

instant_phases = instant_phase(imfs)

instant_freqs = np.diff(instant_phases)/(2*np.pi*dt)
# Create a figure consisting of 3 panels which from the top are the input 

# signal, IMFs and instantaneous frequencies

fig, axes = plt.subplots(3, figsize=(12, 12))



# The top panel shows the input signal

ax = axes[0]

ax.plot(t, S)

ax.set_ylabel("Amplitude [arbitrary units]")

ax.set_title("Input signal")



# The middle panel shows all IMFs

ax = axes[1]

for num, imf in enumerate(imfs):

    ax.plot(t, imf, label='IMF %s' %( num + 1 ))



# Label the figure

#ax.legend()

ax.set_ylabel("Amplitude [arb. u.]")

ax.set_title("IMFs")



# The bottom panel shows all instantaneous frequencies

ax = axes[2]

for num, instant_freq in enumerate(instant_freqs):

    ax.plot(t[:-1], instant_freq, label='IMF %s'%(num+1))



# Label the figure

#ax.legend()

ax.set_xlabel("Time [s]")

ax.set_ylabel("Inst. Freq. [Hz]")

ax.set_title("Huang-Hilbert Transform")



plt.tight_layout()

plt.savefig('Signal-01-Amplitudes', dpi=120)

plt.show()

# Plot results

nIMFs = imfs.shape[0]

plt.figure(figsize=(24,24))

plt.subplot(nIMFs+1, 1, 1)

plt.plot(S, 'r')



for n in range(nIMFs):

    plt.subplot(nIMFs+1, 1, n+2)

    plt.plot(imfs[n], 'g')

    plt.ylabel("IMF %i" %(n+1))

    plt.locator_params(axis='y', nbins=5)



plt.xlabel("Time [s]")

#plt.tight_layout()

plt.savefig('Signal-01', dpi=120)

plt.show()
# The top panel shows the input signal

ax = axes[0]

ax.plot(S)

ax.set_ylabel("Amplitude [arbitrary units]")

ax.set_title("Input signal")

plt.show()
plt.plot(S)

plt.show()