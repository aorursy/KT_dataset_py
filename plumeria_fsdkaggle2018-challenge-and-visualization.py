import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from pyAudioAnalysis import audioTrainTest as aT
#from PyLyrics import *
from wordcloud import WordCloud
from os import path
import os
from scipy.io.wavfile import read
from collections import Counter
import librosa
import librosa.display
import IPython.display
import matplotlib.style as ms
ms.use('seaborn-muted')
audio=pd.read_csv('../input/train.csv')
submission=pd.read_csv('../input/sample_submission.csv')
# Load list of all files from audio_train folder
files_train = librosa.util.find_files('../input/audio_train//')
# Load list of all files from audio_train folder
files_test= librosa.util.find_files('../input/audio_test//')
#y, sr = librosa.load(librosa.util.filename())
fs, data = read('../input/audio_train/001ca53d.wav')
data_size = len(data)
# we will use the size of the array
# to determine the duration of the sound
print(fs)
print(audio.shape)
audio=pd.DataFrame(audio)
print(audio.head(5))
wordcloud = WordCloud(max_font_size=50, width=600, height=300).generate(' '.join(audio.label))
plt.figure(figsize=(15,8))
plt.imshow(wordcloud,interpolation='bilinear')
plt.title("WordCloud for Different Types of Audio Samples", fontsize=35)
plt.axis("off")
plt.show() 
c = Counter(audio.label)
print(list(c))
print('\nThe Audio samples count is {}.'.format(len(audio.label.value_counts())))
# Play the  original audio
y, sr = librosa.load('../input/audio_train//001ca53d.wav')
IPython.display.Audio(data=y, rate=sr)
# Separating harmonic and percussive components
y_h, y_p = librosa.effects.hpss(y)
# Play the harmonic component
IPython.display.Audio(data=y_h, rate=sr)
# Play the percussive component
IPython.display.Audio(data=y_p, rate=sr)
#Plot the amplitude envelope of a waveform.
#If y is monophonic, a filled curve is drawn between [-abs(y), abs(y)].
plt.figure()
plt.subplot(3, 1, 1)
librosa.display.waveplot(y, sr=sr)
plt.title('Monophonic')
# Visualize the MFCC series
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
#Generate mfccs from a time series
librosa.feature.mfcc(y=y, sr=sr)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S,ref=np.max),y_axis='mel', fmax=8000,x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
#Generate melspectrogram from a time series
librosa.feature.melspectrogram(y=y, sr=sr)
#Compare a long-window STFT chromagram to the CQT chromagram
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr,n_chroma=12, n_fft=4096)
chroma_cq = librosa.feature.chroma_cqt(y=y, sr=sr)
plt.figure()
plt.subplot(2,1,1)
librosa.display.specshow(chroma_stft, y_axis='chroma')
plt.title('chroma_stft')
plt.colorbar()
plt.subplot(2,1,2)
librosa.display.specshow(chroma_cq, y_axis='chroma', x_axis='time')
plt.title('chroma_cqt')
plt.colorbar()
plt.tight_layout()
