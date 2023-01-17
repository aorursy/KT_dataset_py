from IPython.display import HTML

HTML('<iframe width="1221" height="687" src="https://www.youtube.com/embed/0ALKGR0I5MA" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting
print(os.listdir('../input/speech-recognition-and-speaker-diarization'))
import IPython.display as ipd  # To play sound in the notebook

fname1 = '../input/speech-recognition-and-speaker-diarization/' + 'meeting-clip2.wav'   # Hi-hat

ipd.Audio(fname1)
import IPython.display as ipd  # To play sound in the notebook

fname2 = '../input/speech-recognition-and-speaker-diarization/' + 'meeting-clip1.mp3'   # Hi-hat

ipd.Audio(fname2)
# Using wave library

import wave

wav = wave.open(fname1)

print("Sampling (frame) rate = ", wav.getframerate())

print("Total samples (frames) = ", wav.getnframes())

print("Duration = ", wav.getnframes()/wav.getframerate())
# Using scipy

from scipy.io import wavfile

rate, data = wavfile.read(fname1)

print("Sampling (frame) rate = ", rate)

print("Total samples (frames) = ", data.shape)
print(data)
plt.plot(data, '-', );
plt.figure(figsize=(16, 4))

plt.plot(data[:500], '.'); plt.plot(data[:500], '-');