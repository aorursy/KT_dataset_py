# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import librosa.display, librosa

import scipy

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



print(os.listdir('/kaggle/input'))

# Any results you write to the current directory are saved as output.



import warnings

warnings.filterwarnings('ignore')
path = '/kaggle/input/mfcc-feature-extraction/mfcc-extraction-wav'

[file for file in os.listdir(path)]
plt.figure(figsize=(15,5))



for i, file in enumerate(os.listdir(path)):

    wav = os.path.join(path,file)



    x, sr = librosa.load(wav)

    #mfcc

    mfccs = librosa.feature.mfcc(x, sr, n_mfcc=13, hop_length=112, n_fft=512, fmin=133.3333, fmax=sr/2, n_mels=40)

    

    plt.subplot(2,5,i+1)

    

    plt.pcolormesh(np.abs(np.log(mfccs)), cmap='gray_r')

    plt.axis('off')
# stft

f, t, y = np.abs(scipy.signal.stft(x, window='hann', nperseg=512, noverlap= 400))
# melfilter

melfilter = librosa.filters.mel(sr, n_fft=512, n_mels=40, fmin=133.333, fmax=sr/2)



# plot filters

plt.figure(figsize=(10,5))

for i in range(melfilter.shape[0]):

   

    plt.plot(melfilter[i, :])
# mel spectrum

melspec = np.dot(melfilter, y)
# cepstral analysis

melspeclog = np.log(melspec)

dct = scipy.fftpack.dct(melspeclog, norm='ortho')

plt.pcolormesh(np.abs(np.log(dct))[:13], cmap='gray_r')