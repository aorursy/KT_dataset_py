# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install pydub
import seaborn as sns
import scipy.io.wavfile
import matplotlib.pyplot as plt
from pydub import AudioSegment as read
from scipy.fftpack import fft,fftfreq
audio = read.from_mp3('/kaggle/input/birdsong-recognition/train_audio/nutwoo/XC462016.mp3')
audio.export("file.wav", format="wav")
print("Listen to the audio clip")
audio
sampling_rate,data = scipy.io.wavfile.read("file.wav")
print("Sampling rate of the audio signal:",sampling_rate)
print("Number of data points:",len(data))
print("Length of the audio clip in seconds:",len(data)/sampling_rate)
print("Energy of the audio signal : {:e}".format(np.sum(data.astype(float)**2)))
time = np.arange(0, float(data.shape[0]), 1) / sampling_rate

plt.figure(figsize=(14, 6))
plt.plot(time,data)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Waveplot')
plt.grid(True)
#Get the absolute value of real and complex components
f_components = abs(fft(data))

# frequencies
freqs = fftfreq(data.shape[0],1/sampling_rate)

plt.figure(figsize=(8, 6))
plt.xlim( [10, sampling_rate/2] )
plt.xscale( 'log' )
plt.grid( True )
plt.xlabel( 'Frequency (Hz)' )
plt.title('FFT of the audio signal')
plt.plot(freqs[:int(freqs.size/2)],f_components[:int(freqs.size/2)])
n = len(data)
f_components = f_components[0:(int(n/2))]

# scale by the number of points so that the magnitude does not depend on the length
f_components = f_components / float(n)

#calculate the frequency at each point in Hz
freqArray = np.arange(0, (n/2), 1.0) * (sampling_rate*1.0/n);

plt.figure(figsize=(8, 6))
plt.plot(freqArray/1000, 10*np.log10(f_components), linewidth=0.1)
plt.get_cmap('autumn_r')
plt.title('Power-Frequency Spectrum')
plt.xlabel('Frequency (kHz)')
plt.ylabel('Power (dB)')
pgram, freqs, bins, im = plt.specgram(data, Fs=sampling_rate, NFFT=1024, cmap=plt.get_cmap('autumn_r'))
cbar=plt.colorbar(im)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
cbar.set_label('Intensity dB')
index = np.where(freqs==12532.32421875)
segment =pgram[index[0][0],:]
plt.plot(bins,segment, color='#ff7f00')
train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
train.head(5)
train.info()
