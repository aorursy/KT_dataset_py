# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import librosa

import IPython.display as ipd

import matplotlib.pyplot as plt

import librosa.display

from matplotlib import gridspec

from PIL import Image

import warnings

warnings.filterwarnings("ignore")
data, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/brdowl/XC413729.mp3",sr=None)

print(type(data))

print(len(data))

print(type(sr))

print(sr)
ipd.Audio("/kaggle/input/birdsong-recognition/train_audio/brdowl/XC413729.mp3")
plt.figure(figsize=(14, 5))

librosa.display.waveplot(data, sr=sr)
#display Spectrogram

X = librosa.stft(data)

Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') 

plt.colorbar()
# log of frequencies 

plt.figure(figsize=(14, 5))

plt.title="Log Spectrom",

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')

plt.colorbar()
# Zooming in Zero Crossing Rate

n0 = 10000

n1 = 10100

plt.figure(figsize=(14, 5))

plt.plot(data[n0:n1])

plt.grid()
zero_crossings = librosa.zero_crossings(data[n0:n1], pad=False)

print(sum(zero_crossings))
#spectral centroid -- centre of mass -- weighted mean of the frequencies present in the sound

import sklearn



spectral_centroids = librosa.feature.spectral_centroid(data, sr=sr)[0]

print(spectral_centroids.shape)



# Computing the time variable for visualization

frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames)



# Normalising the spectral centroid for visualisation

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)



#Plotting the Spectral Centroid along the waveform

plt.figure(figsize=(14, 5))

librosa.display.waveplot(data, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_centroids), color='r')
plt.figure(figsize = (14,5))

spectral_rolloff = librosa.feature.spectral_rolloff(data, sr=sr)[0]

librosa.display.waveplot(data, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_rolloff), color='r')
mfccs = librosa.feature.mfcc(data, sr=sr)

print(mfccs.shape)

#Displaying  the MFCCs:

plt.figure(figsize = (14,5))

librosa.display.specshow(mfccs, sr=sr, x_axis='time')
fig = plt.figure(figsize=(18, 18)) 

gs = gridspec.GridSpec(5, 2, width_ratios=[2, 6]) 

ax0 = plt.subplot(gs[0])

img = Image.open("/kaggle/input/osic-bird-image/bkbwar.jpg")

ax0.axis('off')

ax0.imshow(img)

ax1 = plt.subplot(gs[1])

y, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/bkbwar/XC101580.mp3")

C = librosa.feature.chroma_cqt(y=y, sr=sr)

librosa.display.specshow(C, y_axis='chroma', x_axis='time')

ax1.plot()



ax2 = plt.subplot(gs[2])

img = Image.open("/kaggle/input/osic-bird-image/clanut.jpg")

ax2.axis('off')

ax2.imshow(img)

ax3 = plt.subplot(gs[3])

y1, sr1 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/clanut/XC391597.mp3")

C1 = librosa.feature.chroma_cqt(y=y1, sr=sr1)

librosa.display.specshow(C1, y_axis='chroma', x_axis='time')

ax3.plot()

   

ax4 = plt.subplot(gs[4])

img = Image.open("/kaggle/input/osic-bird-image/whcspa.jpg")

ax4.axis('off')

ax4.imshow(img)

ax5 = plt.subplot(gs[5])

y2, sr2 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/whcspa/XC478423.mp3")

C2 = librosa.feature.chroma_cqt(y=y2, sr=sr2)

librosa.display.specshow(C2, y_axis='chroma', x_axis='time')

ax5.plot()



ax6 = plt.subplot(gs[6])

img = Image.open("/kaggle/input/osic-bird-image/prawar.jpg")

ax6.axis('off')

ax6.imshow(img)

ax7 = plt.subplot(gs[7])

y3, sr3 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/prawar/XC444966.mp3")

C3 = librosa.feature.chroma_cqt(y=y3, sr=sr3)

librosa.display.specshow(C3, y_axis='chroma', x_axis='time')

ax7.plot()



ax8 = plt.subplot(gs[8])

img = Image.open("/kaggle/input/osic-bird-image/rebwoo.jpg")

ax8.axis('off')

ax8.imshow(img)

ax9 = plt.subplot(gs[9])

y4, sr4 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/rebwoo/XC145839.mp3")

C4 = librosa.feature.chroma_cqt(y=y4, sr=sr4)

librosa.display.specshow(C4, y_axis='chroma', x_axis='time')

ax9.plot()
fig = plt.figure(figsize=(15, 13)) 

gs = gridspec.GridSpec(5, 2, width_ratios=[2, 6]) 

ax0 = plt.subplot(gs[0])

img = Image.open("/kaggle/input/osic-bird-image/bkbwar.jpg")

ax0.axis('off')

ax0.imshow(img)

ax1 = plt.subplot(gs[1])

y, sr = librosa.load("/kaggle/input/birdsong-recognition/train_audio/bkbwar/XC101580.mp3")

librosa.display.waveplot(y, sr)

ax1.plot()



ax2 = plt.subplot(gs[2])

img = Image.open("/kaggle/input/osic-bird-image/clanut.jpg")

ax2.axis('off')

ax2.imshow(img)

ax3 = plt.subplot(gs[3])

y1, sr1 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/clanut/XC391597.mp3")

librosa.display.waveplot(y1, sr1)

ax3.plot()

   

ax4 = plt.subplot(gs[4])

img = Image.open("/kaggle/input/osic-bird-image/whcspa.jpg")

ax4.axis('off')

ax4.imshow(img)

ax5 = plt.subplot(gs[5])

y2, sr2 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/whcspa/XC478423.mp3")

librosa.display.waveplot(y2, sr2)

ax5.plot()



ax6 = plt.subplot(gs[6])

img = Image.open("/kaggle/input/osic-bird-image/prawar.jpg")

ax6.axis('off')

ax6.imshow(img)

ax7 = plt.subplot(gs[7])

y3, sr3 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/prawar/XC444966.mp3")

librosa.display.waveplot(y3, sr3)

ax7.plot()



ax8 = plt.subplot(gs[8])

img = Image.open("/kaggle/input/osic-bird-image/rebwoo.jpg")

ax8.axis('off')

ax8.imshow(img)

ax9 = plt.subplot(gs[9])

y4, sr4 = librosa.load("/kaggle/input/birdsong-recognition/train_audio/rebwoo/XC145839.mp3")

librosa.display.waveplot(y4, sr4)

ax9.plot()