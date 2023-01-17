# Some info re. this iPython notebook:
# -----------------------------------
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
# -----------------------------------
# Import some audio and visualisation libraries
import librosa
import librosa.display
import matplotlib.pyplot as plt
# 1 - Import and load the wav file (using librosa)
y, sr = librosa.load('../input/rID-1_speaker-1_trial-1.wav')
t = np.arange(0,len(y))/sr

# Have a look!
f0 = plt.figure(figsize=(12,4))
plt.plot(t,y)
plt.title('rID-1_speaker-1_trial-1.wav')
plt.xlabel('time (s)')
plt.ylabel('amplitude')

# Compute mel-scaled spectrogram, and scale to dB
s = librosa.feature.melspectrogram(y=y, sr=sr)
sdB = librosa.power_to_db(s,ref=np.max)
# Import target words list and ground truth
targets = np.genfromtxt('../input/rID-1_speaker-1_trial-1_targets.txt',delimiter=' ',dtype=str)
groundtruth = np.genfromtxt('../input/rID-1_speaker-1_trial-1_ground_truth.txt',delimiter=' ',dtype=str)
# 2 - Estimate word locations
# > Take a look at the power in frequencies below 512 Hz, covering the resonant freqs of human speech
p512 = (sdB[0:31,:]).mean(axis=0)
# > threshold this mean above -45 dB, to estimate where speech is found
mask=np.where(p512>-45,1,0)

# A crude partitioning algorithm
transitions=[]
mask2 = np.append(np.insert(mask,0,1),1) # add '1' at start and end
for i in range(len(mask2)-1):
    if (mask2[i] == 1) & (mask2[i+1] == 0):
        transitions.append(i)
    elif (mask2[i] == 0) & (mask2[i+1] == 1):
        transitions.append(i)
# > generates the locations where 1s become 0s and vice versa
silence = np.array(list(zip(transitions,transitions[1:])))
# > split these intervals of silence
pauses = [x.mean() for x in silence if np.diff(x) > 40]
# > convert to nearest integers array
pauses = np.array(pauses).astype(int)


# Have a look!
f1 = plt.subplots(2,1,sharex=True,figsize=(12,12))
ax1=plt.subplot(3,1,1)
plt.plot(p512,'m')
plt.plot(pauses,p512[pauses],'r.')
plt.title("Mean mel-binned power below 512Hz")
plt.ylabel('dB')

ax2=plt.subplot(3,1,2)
mask=np.where(p512>-45,1,0)
plt.plot(mask,'g')
plt.plot(pauses,mask[pauses],'r.')
plt.title("Speech indicator boolean")
plt.ylabel('boolean')

ax2=plt.subplot(3,1,3)
plt.plot(t,y)
for xc in pauses*t.max()/sdB.shape[1]:
    plt.axvline(x=xc,color='r',linestyle=':')
plt.title("Partitioned audio stream")
plt.xlabel('time (s)')
plt.ylabel('amplitude')

plt.show()

# 3 - Now that we have partitioned the audiostream, let's have a look at a couple words up close, in their captured
# representation as air pressure perturbation amplitudes

# Plot some words
# > compute indices for down-sampled mel-spectrum and full time domain
word_times_m = list(zip(pauses,pauses[1:]))
word_times_t = np.array(word_times_m)*t.max()/sdB.shape[1]


# Prepare figure
f2, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4), sharey=True)
ax1 = plt.subplot(1,2,1)
plt.ylabel('amplitude')
plt.xlabel('time (s)')

plt.plot(t,y)
# Zoom in on first word
ax1.set_xlim([word_times_t[0,0], word_times_t[0,1]])
# Adjust plot 
plt.title("Waveform of '{}'".format(groundtruth[0]))
plt.tight_layout()

ax2 = plt.subplot(1,2,2)
plt.plot(t,y)
# Zoom in on second word
ax2.set_xlim([word_times_t[1,0], word_times_t[1,1]])
# Adjust plot 
plt.title("Waveform of '{}'".format(groundtruth[1]))
plt.tight_layout()
plt.xlabel('time (s)')

plt.show()
# 4 - Now let's have a look at what these words look like in the mel-spectrogram representation
# The mel-spectrogram is a graphical representation of the power spectral densities of the waveform at successive timebins.
# The 'mel' prefix refers to a scaling of the frequency bins into roughly equal energy -- ie, more bins at lower frequencies,
# and fewer bins at higher ones.


# Prepare figure
f3, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4), sharey=True)

# Plot some words
ax1 = plt.subplot(1,2,1)
librosa.display.specshow(sdB, y_axis='mel', fmax=8000, x_axis='time')
# Zoom in on first word
ax1.set_xlim([word_times_t[0,0], word_times_t[0,1]])
# Adjust plot 
plt.colorbar(format='%+2.0f dB')
plt.title("Mel spectrogram for '{}'".format(groundtruth[0]))
plt.tight_layout()

ax2 = plt.subplot(1,2,2)
librosa.display.specshow(sdB, y_axis='mel', fmax=8000, x_axis='time')
# Zoom in on second word
ax2.set_xlim([word_times_t[1,0], word_times_t[1,1]])
# Adjust plot 
plt.colorbar(format='%+2.0f dB')
plt.title("Mel spectrogram for '{}'".format(groundtruth[1]))
plt.tight_layout()

plt.show()