import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from scipy.io import wavfile

import librosa

import librosa.display



import warnings

warnings.filterwarnings('ignore')
# simple sine wave

A = 1                      # Amplitude

f0 = 1                     # frequency

Fs = 1600                  # Sampling frequency

t = np.arange(0, 1, 1/Fs) 



X = A * np.sin(2 * np.pi * f0 * t)



plt.plot(t, X)

plt.xlabel("Time")

plt.ylabel("Amplitude")

plt.show()
# create two independent waves then adding them resulting wave is complex wave.



# wave 1

A = 1/3

f0 = 20

Fs = 1600

t = np.arange(0, 1, 1/Fs)

X1 = A * np.sin(2  * np.pi * f0 * t)



# wave 2

A = 0.5

f0 = 3

Fs = 1600

t = np.arange(0, 1, 1/Fs)

X2 = A * np.cos(2  * np.pi * f0 * t)



# wave 3 = wave 1 + wave 2

X3 = X1 + X2





plt.figure(figsize=(10,5))

plt.plot(X1)

plt.plot(X2)

plt.plot(X3)

plt.legend(['x1','x2','x3'])

plt.show()
# read speech audio file using scipy package



wav_path = "../input/birdsong-recognition/example_test_audio/BLKFR-10-CPL_20190611_093000.pt540.mp3"

data, fs = librosa.load(wav_path)

# fs is sampling frequency

# sampling frequency nothing but how may samples present for second.

print(f"Sampling frequency : {fs} and Wave : {data}")
plt.plot(data)

plt.xlabel("Time")

plt.ylabel("Amplitude")

plt.show()
# https://www.ritchievink.com/blog/2017/04/23/understanding-the-fourier-transform-by-example/



# Example

# From complex wave convert Time Domine to Frequency Domine using FFT





t = np.linspace(0, 0.5, 500)

s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)



plt.ylabel("Amplitude")

plt.xlabel("Time [s]")

plt.plot(t, s)

plt.show()
fft = np.fft.fft(s)

T = t[1] - t[0]  # sampling interval 

N = s.size



# 1/T = frequency

f = np.linspace(0, 1 / T, N)



plt.ylabel("Amplitude")

plt.xlabel("Frequency [Hz]")

plt.bar(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, width=1.5)  # 1 / N is a normalization factor

plt.show()
y, sr = librosa.load(wav_path)

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)



plt.figure(figsize=(10,4))

librosa.display.specshow(mfccs, x_axis="time")

plt.colorbar()

plt.title('MFCC')

plt.tight_layout()

plt.show()