import librosa

import numpy as np

import IPython.display as ipd

import matplotlib.pyplot as plt
x, sr = librosa.load('../input/freesound-audio-tagging-2019/train_curated/f5342540.wav', sr=44100)

x = x[:len(x) // 7]
np.random.seed(2019)

PDM = np.array([np.complex(np.cos(p), np.sin(p)) for p in np.random.normal(0, 0.4, size=513)])
f = librosa.stft(x, window='hanning', hop_length=512, n_fft=1024)

f *= PDM.reshape((-1, 1))
y = librosa.istft(f, window='hanning', hop_length=512)

x = x[:len(y)]
plt.figure(figsize=(15, 10))



plt.subplot(2, 1, 1)

plt.plot(x)



plt.subplot(2, 1, 2)

plt.plot(y)



plt.show()
ipd.Audio(x, rate=sr)
ipd.Audio(y, rate=sr)