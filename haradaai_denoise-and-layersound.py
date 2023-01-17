!pip install noisereduce
import librosa
import librosa.display
import random
import IPython

import numpy as np
import pandas as pd
import noisereduce as nr

from pathlib import Path
from matplotlib import pyplot as plt
from scipy.ndimage import maximum_filter1d
def view_audio(audio_path,swich = [1,1,1]):
    y, sr = librosa.load(audio_path)

    graphs = len(swich)

    fig = plt.figure(figsize=(20,graphs*5))
    
    if swich[0] == 1:
        ax1 = fig.add_subplot(graphs,1,1,title='waveplot')
        # 波形で表示
        librosa.display.waveplot(y, sr=sr)
    
    if swich[1] == 1:
        # メルスペクトログラムとやらに変換
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        # デシベル（音量）スケールのスペクトログラムに変換
        log_S = librosa.amplitude_to_db(S, ref=np.max)

        # librosaのスペクトログラムを出してくれるAPIを呼ぶ
        ax2 = fig.add_subplot(graphs,1,2,title='mel power spectrogram')
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar()
    
    if swich[2] ==1:        
        # 短時間フーリエ変換 
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        ax3 = fig.add_subplot(graphs,1,3,title='stft')
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
    
    fig.show()
audio_path = '/kaggle/input/birdsong-recognition/train_audio/whtswi/XC425114.mp3'
#ipd.Audio(audio_path)

view_audio(audio_path)
def envelope(y, rate, threshold):
    mask = []
    y_mean = maximum_filter1d(np.abs(y), size=rate//20)
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask, y_mean
TEST_DIR = '../input/birdcall-check/test_audio'
test_path = []
import os
for dirname, _, filenames in os.walk(TEST_DIR):
    for filename in filenames:
        audio_path = os.path.join(dirname, filename)
        test_path.append(audio_path)


print(test_path)
import warnings
warnings.filterwarnings('ignore')
thr = 0.25
x_deonoise = []


for i in range(len(test_path)):
    x, sr = librosa.load(test_path[i])
    mask, env = envelope(x, sr, thr)
    x_deonoise.append(nr.reduce_noise(audio_clip=x, noise_clip=x[np.logical_not(mask)], verbose=False))

## blue →(denoise)→ orenge
for i in range(len(test_path)):
    x, sr = librosa.load(test_path[i])
    plt.plot(x)
    plt.plot(x_deonoise[i])

    plt.show()
x,sr = librosa.load('../input/birdcall-check/test_audio/856b194b097441958697c2bcd1f63982.mp3')
# 元の音声
IPython.display.Audio(data=x, rate=sr)
# ノイズ除去
IPython.display.Audio(data=x_deonoise[0], rate=sr)
ald_0 , str_0 = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3',offset=0,duration=5)
ald_1 , str_1 = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC135454.mp3',offset=0,duration=5)
def view_audio(y,sr,swich = [1,1,1]):
    
    graphs = len(swich)

    fig = plt.figure(figsize=(20,graphs*5))
    
    if swich[0] == 1:
        ax1 = fig.add_subplot(graphs,1,1,title='waveplot')
        # 波形で表示
        librosa.display.waveplot(y, sr=sr)
    
    if swich[1] == 1:
        # メルスペクトログラムとやらに変換
        S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
        # デシベル（音量）スケールのスペクトログラムに変換
        log_S = librosa.amplitude_to_db(S, ref=np.max)

        # librosaのスペクトログラムを出してくれるAPIを呼ぶ
        ax2 = fig.add_subplot(graphs,1,2,title='mel power spectrogram')
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar()
    
    if swich[2] ==1:        
        # 短時間フーリエ変換 
        X = librosa.stft(y)
        Xdb = librosa.amplitude_to_db(abs(X))
        ax3 = fig.add_subplot(graphs,1,3,title='stft')
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
    
    fig.show()
IPython.display.Audio(data=ald_0, rate=str_0)
view_audio(ald_0,str_0,[1,0,0])
IPython.display.Audio(data=ald_1, rate=str_1)
view_audio(ald_1,str_1,[1,0,0])
IPython.display.Audio(data=ald_0 +ald_1, rate=str_0)
view_audio(ald_0+ald_1,str_0,[1,0,0])
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+1e-6)
    return data-0.5
norm_0 = audio_norm(ald_0)
norm_1 = audio_norm(ald_1)
IPython.display.Audio(data=norm_0 +norm_1, rate=str_0)
view_audio(norm_0+norm_1,str_0,[1,0,0])
x = norm_0+norm_1
mask, env = envelope(x, str_0, thr)
denoise = nr.reduce_noise(audio_clip=x, noise_clip=x[np.logical_not(mask)], verbose=False)

plt.plot(x)
plt.plot(denoise)
plt.show()
