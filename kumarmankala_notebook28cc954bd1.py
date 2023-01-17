# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from numpy import sin, linspace, pi

from pylab import plot, show, title, xlabel, ylabel, subplot

from scipy import fft, arange



def plotSpectrum(y,Fs):

 """

 Plots a Single-Sided Amplitude Spectrum of y(t)

 """

 n = len(y) # length of the signal

 k = arange(n)

 T = n/Fs

 frq = k/T # two sides frequency range

 frq = frq[range(n/2)] # one side frequency range



 Y = fft(y)/n # fft computing and normalization

 Y = Y[range(n/2)]

 

 plot(frq,abs(Y),'r') # plotting the spectrum

 xlabel('Freq (Hz)')

 ylabel('|Y(freq)|')



Fs = 150.0;  # sampling rate

Ts = 1.0/Fs; # sampling interval

t = arange(0,1,Ts) # time vector



ff = 5;   # frequency of the signal

y = sin(2*pi*ff*t)