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

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pdb

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Lambda, Activation

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras import backend as K

from matplotlib import pyplot as plt

%matplotlib inline

import scipy

from scipy import misc, ndimage

from scipy.ndimage.interpolation import zoom

from scipy.ndimage import imread





train = pd.read_json('../input/train.json')
def get_images(df):

    '''Create 3-channel 'images'. Return rescale-normalised images.'''

    images = []

    for i, row in df.iterrows():

        # Formulate the bands as 75x75 arrays

        band_1 = np.array(row['band_1']).reshape(75, 75)

        band_2 = np.array(row['band_2']).reshape(75, 75)

        band_3 = band_1 / band_2



        # Rescale

        r = (band_1 - band_1.min()) / (band_1.max() - band_1.min())

        g = (band_2 - band_2.min()) / (band_2.max() - band_2.min())

        b = (band_3 - band_3.min()) / (band_3.max() - band_3.min())



        rgb = np.dstack((r, g, b))

        images.append(rgb)

    return np.array(images)
X = get_images(train)

y = to_categorical(train.is_iceberg.values,num_classes=2)
Xtr, Xv, ytr, yv = train_test_split(X, y, shuffle=False, test_size=0.20)
# Xtr[0]
from scipy.fftpack import rfft

from scipy.signal import correlate, resample, welch



%matplotlib inline

# autoreload class definition

%load_ext autoreload

%autoreload 2



import matplotlib.pyplot as plt

#1_121_1.mat 

x=Xtr[1].ravel()

x=x.transpose()

x = resample(x, 300, axis=0, window=400)



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



signal=Xtr[0].ravel()

#signal=remove_dc(signal)

signal=blur(signal)

n  = len(signal)      # Get the signal length

dt = 1/float(2400) # Get time resolution

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



signal=Xtr[1].ravel()

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



x_train=Xtr[1].ravel()



N=100



Hn = np.fft.fft(x_train)

f = np.fft.fftfreq(N)



ind = np.arange(1,N//2+1) # Need integer division!



psd = np.abs(Hn[ind])**2 + np.abs(Hn[-ind])**2

print (len(psd))

plt.plot(f[ind], psd, 'k-')

plt.xlim(xmax=5, xmin=-5)



temp = np.partition(-psd, 15)

print (len(temp))

x_psd = -temp[:32]

print (len(x_psd))

print ('Top PSD peaks:' + str(x_psd))
np.set_printoptions(precision=4, threshold=10000, linewidth=100, edgeitems=999, suppress=True)



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

pd.set_option('display.width', 100)

pd.set_option('expand_frame_repr', False)

pd.set_option('precision', 6)





#%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import matplotlib.mlab as mlab



x_train=Xtr[0]

print (x_train.shape)



# sampling rate:

fs = 240000

# number of sample for the fast fourier transform:

NFFT = 1*fs

fmin = 10

fmax = 20000

Pxx_H1, freqs = mlab.psd(x_train.T, Fs = fs, NFFT = NFFT)

psd_H1 = interp1d(freqs, Pxx_H1)



plt.figure()

plt.loglog(freqs, np.sqrt(Pxx_H1),'r',label='Train')

plt.axis([fmin, fmax, 1e-24, 1e-1])

plt.grid('on')

plt.ylabel('rtHz)')

plt.xlabel('Freq (Hz)')

plt.legend(loc='upper center')

plt.title('Freq Plot')
#print("Dominant frequencies:", f[ind[np.where(psd>0.3e20)]]) # SET LIMIT BY HAND

#print("True frequencies:    ", fs)