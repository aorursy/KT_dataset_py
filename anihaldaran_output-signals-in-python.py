# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd

main_file_path = '../input/dataset_no_outlier.csv' # this is the path to the genetic data that you will use
data = pd.read_csv(main_file_path)
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

fs=105e6
fin=70.1e6

N=np.arange(0,21e3,1)

# Create a input sin signal of 70.1 MHz sampled at 105 MHz
x_in=np.sin(2*np.pi*(fin/fs)*N)

# Define the "b" and "a" polynomials to create a CIC filter (R=8,M=2,N=6)
b=data.X
b[[0,16,32,48,64,80,96]]=[1,-6,15,-20,15,-6,1]
a=data.Y
a[[0,1,2,3,4,5,6]]=[1,-6,15,-20,15,-6,1]

w,h=signal.freqz(b,a)
plt.plot(w/max(w),20*np.log10(abs(h)/np.nanmax(h)))
plt.title('CIC Filter Response')

output_nco_cic=signal.lfilter(b,a,x_in)

plt.figure()        
plt.plot(x_in)
plt.title('Input Signal')
plt.figure()        
plt.plot(output_nco_cic)
plt.title('Filtered Signal')
# import the libraries

import matplotlib.pyplot as plot

import numpy as np

 

# Define the list of frequencies

frequencies         = np.arange(5,105,5)

 

# Sampling Frequency

samplingFrequency   = 400

 

# Create two ndarrays

s1 = data.X # For samples

s2 = data.Mixed # For signal

 

# Start Value of the sample

start   = 1

 

# Stop Value of the sample

stop    = samplingFrequency+1

 

for frequency in frequencies:

    sub1 = np.arange(start, stop, 1)

 

    # Signal - Sine wave with varying frequency + Noise

    sub2 = np.sin(2*np.pi*sub1*frequency*1/samplingFrequency)+np.random.randn(len(sub1))

  

    s1      = np.append(s1, sub1)

    s2      = np.append(s2, sub2)

   

    start   = stop+1

    stop    = start+samplingFrequency

 

# Plot the signal

plot.subplot(211)

plot.plot(s1,s2)

plot.xlabel('Sample')

plot.ylabel('Amplitude')

 

 

# Plot the spectrogram

plot.subplot(212)

powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(s2, Fs=samplingFrequency)

plot.xlabel('Time')

plot.ylabel('Frequency')

 

plot.show()   