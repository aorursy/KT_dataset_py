%reset -f

__author__ = 'Solomonk: https://www.kaggle.com/solomonk/'



# Standard python numerical analysis imports:

import numpy as np

from scipy import signal

from scipy.interpolate import interp1d

from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz



import scipy.io

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import numpy

import pandas as pd

import numpy

import pandas

import os

from scipy.io import loadmat

from PIL import Image

import numpy as np



#--------------------DEFINE DATA IEEG SETS-----------------------#

DATA_FOLDER= '../input/train_3/'



#SINGLE MAT FILE FOR EXPLORATION

TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE=DATA_FOLDER + "3_100_1.mat"

#TEST_1_DATA_FOLDER_IN_SINGLE_FILE=DATA_FOLDER + "/1_102_1.mat"



#--------------------DEFINE DATA SETS-----------------------#





#---------------------------------------------------------------#

def ieegMatToPandasDF(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return pd.DataFrame(ndata['data'], columns=ndata['channelIndices'][0])   



def ieegMatToArray(path):

    mat = loadmat(path)

    names = mat['dataStruct'].dtype.names

    ndata = {n: mat['dataStruct'][n][0, 0] for n in names}

    return ndata['data']  



#---------------------------------------------------------------#



#---------------------------------------------------------------#

def ieegSingleMetaData(path):

    mat_data = scipy.io.loadmat(path)

    data = mat_data['dataStruct']

    for i in [data, data[0], data[0][0][0], data[0][0][0][0]]:

        print((i.shape, i.size))

#---------------------------------------------------------------#        



#---------------------------------------------------------------#

def ieegGetFilePaths(directory, extension='.mat'):

    filenames = sorted(os.listdir(directory))

    files_with_extension = [directory + '/' + f for f in filenames if f.endswith(extension) and not f.startswith('.')]

    return files_with_extension

#---------------------------------------------------------------#



#---------------------------------------------------------------#

# EEG clips labeled "Preictal" (k=1) for pre-seizure data segments, 

# or "Interictal" (k-0) for non-seizure data segments.

# I_J_K.mat - the Jth training data segment corresponding to the Kth 

# class (K=0 for interictal, K=1 for preictal) for the Ith patient (there are three patients).

def ieegIsInterictal(name):  

    try:

        return float(name[-5])

    except:

        return 0.0

#---------------------------------------------------------------#

ieegSingleMetaData(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)     
from scipy.fftpack import rfft

from scipy.signal import correlate, resample, welch



%matplotlib inline

# autoreload class definition

%load_ext autoreload

%autoreload 2



import matplotlib.pyplot as plt

#1_121_1.mat 

x=ieegMatToArray(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)

x=x.transpose()

x = resample(x, 600, axis=1, window=400)



print((x.shape, x.size))

matplotlib.rcParams['figure.figsize'] = (20.0, 20.0)



# plot each channel 

plt.figure(figsize=(10,4))

plt.plot(x)

plt.xlabel('time')

plt.ylabel('magnitude')

plt.title('%d channels'%x.shape[0])

plt.grid(True)
from scipy import ndimage 

def blur( x):

    img = ndimage.gaussian_filter(x, sigma=(8), order=0)

    return img

def remove_dc(x):

    # print x.shape

    assert (type(x) == np.ndarray)    

    x_dc = np.zeros(x.shape)

    for i in range(x.shape[0]):

        x_dc[i, :] = x[i, :] - np.mean(x[i, :])

    return x_dc



%matplotlib inline

import scipy

import scipy.fftpack

import pylab

from scipy import pi

#http://stackoverflow.com/questions/9456037/scipy-numpy-fft-frequency-analysis



signal=ieegMatToArray(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)

#signal=remove_dc(signal)

signal=blur(signal)

n  = len(signal)      # Get the signal length

dt = 1/float(240000) # Get time resolution

FFT = abs(scipy.fft(signal))

freqs = scipy.fftpack.fftfreq(n, dt)



#pylab.subplot(211)

pylab.plot(signal)

#pylab.subplot(212)

#pylab.plot(freqs,20*scipy.log10(FFT),'x')

pylab.show()

%matplotlib inline

import scipy

import scipy.fftpack

import pylab

from scipy import pi

#http://stackoverflow.com/questions/9456037/scipy-numpy-fft-frequency-analysis



signal=ieegMatToArray(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)

n  = len(signal)      # Get the signal length

dt = 1/float(240000) # Get time resolution

FFT = abs(scipy.fft(signal))

freqs = scipy.fftpack.fftfreq(n, dt)



#pylab.subplot(211)

pylab.plot(signal)

#pylab.subplot(212)

#pylab.plot(freqs,20*scipy.log10(FFT),'x')

pylab.show()
%matplotlib inline



x_train=ieegMatToArray(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)



Hn = np.fft.fft(x_train)

f = np.fft.fftfreq(240000)

#print(f)

N=240000

ind = np.arange(1,N//2+1) # Need integer division!

#print(f[ind])

#print(f[-ind])



psd = np.abs(Hn[ind])**2 + np.abs(Hn[-ind])**2

print (len(psd))

plt.plot(f[ind], psd, 'k-')

plt.xlim(xmax=5, xmin=-5)



temp = np.partition(-psd, 15)

print (len(temp))

x_psd = -temp[:32]

print (len(x_psd))



#print ('Top PSD peaks:' + str(x_psd))
np.set_printoptions(precision=4, threshold=10000, linewidth=100, edgeitems=999, suppress=True)



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.width', 100)

pd.set_option('expand_frame_repr', False)

pd.set_option('precision', 6)





#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab



x_train=ieegMatToArray(TRAIN_1_DATA_FOLDER_IN_SINGLE_FILE)

print (x_train.shape)

x_test=ieegMatToArray(TEST_1_DATA_FOLDER_IN_SINGLE_FILE)

print (x_test.shape)



# sampling rate:

fs = 240000

# number of sample for the fast fourier transform:

NFFT = 1*fs

fmin = 10

fmax = 20000

Pxx_H1, freqs = mlab.psd(x_train.T, Fs = fs, NFFT = NFFT)

Pxx_L1, freqs = mlab.psd(x_test.T, Fs = fs, NFFT = NFFT)





psd_H1 = interp1d(freqs, Pxx_H1)

psd_L1 = interp1d(freqs, Pxx_L1)



plt.figure()

plt.loglog(freqs, np.sqrt(Pxx_H1),'r',label='Train')

plt.loglog(freqs, np.sqrt(Pxx_L1),'g',label='Test')

plt.axis([fmin, fmax, 1e-24, 1e-1])

plt.grid('on')

plt.ylabel('rtHz)')

plt.xlabel('Freq (Hz)')

plt.legend(loc='upper center')

plt.title('Freq Plot')
#print("Dominant frequencies:", f[ind[np.where(psd>0.3e20)]]) # SET LIMIT BY HAND

#print("True frequencies:    ", fs)