! apt install -y libsndfile1
import IPython

import librosa                    

import librosa.display

from scipy.io import wavfile

import matplotlib.pyplot as plt

import numpy as np
sampling_freq, noisy_data = wavfile.read("../input/thesuphichallenge/suphi_noise.wav")

noisy_L = noisy_data[:,0]; noisy_R = noisy_data[:,1]
fig, ax = plt.subplots(figsize=(20,3))

ax.plot(noisy_data)

IPython.display.Audio(data=[noisy_data[:,0], noisy_data[:,1]], rate=sampling_freq)
noisy_data_mel = librosa.feature.melspectrogram(y=noisy_data[:,0].astype(float), sr=sampling_freq, n_mels=128)

noisy_data_mel_dB = librosa.power_to_db(noisy_data_mel, ref=np.max)

plt.figure(figsize=(10, 5))

librosa.display.specshow(noisy_data_mel_dB, x_axis='time', y_axis='mel', sr=sampling_freq, fmax=20000)

plt.colorbar(format='%+1.0f dB')

plt.title("MEL Spectogram of Noisy Data")

plt.tight_layout()

plt.show()
import pandas as pd



df = pd.DataFrame(noisy_data)

df.to_csv("output.csv", header=["Left", "Right"], index=True, index_label="SampleID")