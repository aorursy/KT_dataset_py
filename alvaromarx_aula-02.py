# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import soundfile as sf

import librosa.feature

import librosa.display

import seaborn as sns

import IPython.display as ipd

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import matplotlib.pyplot as plt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
hiphop, sr_hiphop = sf.read("/kaggle/input/gtzan-genre-collection/genres/hiphop/hiphop.00033.au")

classic, sr_classic = sf.read("/kaggle/input/gtzan-genre-collection/genres/classical/classical.00023.au")





ipd.Audio(hiphop,rate=sr_hiphop)
ipd.Audio(classic,rate=sr_classic)
classic_s = classic[sr_classic*4:sr_classic*6]

ipd.Audio(classic_s, rate=sr_classic)
hiphop_s = hiphop[sr_hiphop*4:sr_hiphop*6]

ipd.Audio(hiphop_s, rate=sr_hiphop)
plt.plot(hiphop_s)

plt.show()
plt.plot(classic_s)

plt.show()


centroids_classic = librosa.feature.spectral_centroid(classic_s)

centroids_hiphop = librosa.feature.spectral_centroid(hiphop_s)

plt.plot(centroids_classic[0],label='classic')

plt.plot(centroids_hiphop[0],label='hiphop')

plt.legend()

plt.show()

for i in (classic_s, hiphop_s):

    mfcc = librosa.feature.mfcc(y=i,sr=samplerate,n_mfcc=40)

    melspec = librosa.feature.melspectrogram(y=i,sr=samplerate)

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(mfcc,sr=samplerate,x_axis='time')

    plt.colorbar()

    plt.show()

    plt.figure(figsize=(10, 4))

    librosa.display.specshow(melspec,sr=samplerate,x_axis='time',y_axis='mel',cmap='jet')

    plt.colorbar()

    plt.show()
rms_classic = librosa.feature.rms(y=classic_s)

rms_hiphop = librosa.feature.rms(y=hiphop_s)

plt.plot(rms_classic[0],label='classic')

plt.plot(rms_hiphop[0],label='hiphop')

plt.legend()

plt.show()
flatness_hiphop = librosa.feature.spectral_flatness(y=hiphop_s)

flatness_classic = librosa.feature.spectral_flatness(y=classic_s)

plt.plot(flatness_hiphop[0],label='hiphop_s')

plt.plot(flatness_classic[0],label='classic_s')

plt.legend()

plt.show()

plt.plot(flatness_hiphop[0],label='hiphop_s')

plt.title('hiphop_s')

plt.show()

plt.plot(flatness_classic[0],label='classic_s')

plt.title('classic_s')

plt.show()

rolloff_hiphop = librosa.feature.spectral_rolloff(y=hiphop_s)

rolloff_classic = librosa.feature.spectral_rolloff(y=classic_s)

plt.plot(rolloff_hiphop[0],label='hiphop')

plt.plot(rolloff_classic[0],label='classic')

plt.legend()

plt.show()

windows_classic = np.array_split(classic,30)

windows_hiphop = np.array_split(hiphop, 30)
feat1,feat2 = librosa.feature.rms, librosa.feature.spectral_centroid



values = np.zeros((1,3))



for window_classic,window_hiphop in zip(windows_classic,windows_hiphop): 

    ft1_cl = feat1(window_classic).mean()

    ft1_hh = feat1(window_hiphop).mean()

    ft2_cl = feat2(window_classic).mean()

    ft2_hh = feat2(window_hiphop).mean()

    values = np.vstack([values, np.array([ft1_cl,ft2_cl,0]),np.array([ft1_hh,ft2_hh,1])])

plot_values = values[:,:-1]

label_values = ['hiphop' if value else 'classical' for value in values[:,-1]]

sns.scatterplot(x=plot_values[:,0],y=plot_values[:,1],hue=label_values)