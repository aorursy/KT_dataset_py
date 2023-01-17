import librosa

import matplotlib.pyplot as plt

import librosa.display

import sklearn

import numpy as np
audio_path = "/kaggle/input/audio-data/The Big Bang Theory Season 6 Ep 21 - Best Scenes.wav"
x , sr = librosa.load(audio_path)
#waveform plot

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x, sr=sr)
mfccs = librosa.feature.mfcc(x, sr=sr)

print(mfccs.shape)

mfccs_mean = np.mean(librosa.feature.mfcc(x, sr))

print("MFCCS mean value:",mfccs_mean)
#Displaying  the MFCCs:

librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# Zooming in

n0 = 9000

n1 = 9100

plt.figure(figsize=(14, 5))

plt.plot(x[n0:n1])

plt.xlabel("Time")

plt.ylabel("Magnitude")

plt.grid()
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)

print(sum(zero_crossings))
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

print(spectral_centroids)

print("Mean Spectral Centroid",np.mean(spectral_centroids))
# Computing the time variable for visualization

frames = range(len(spectral_centroids))

t = librosa.frames_to_time(frames)

# Normalising the spectral centroid for visualisation

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)

#Plotting the Spectral Centroid along the waveform

plt.figure(figsize=(60, 20))

plt.xlabel("Time")

plt.ylabel("Magnitude")

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, normalize(spectral_centroids), color='r')
pitches, magnitudes = librosa.piptrack(x,sr)

print(pitches)

#plt.subplot(212)

#plt.show()

print("Mean Pitch",np.mean(pitches))

plt.plot(pitches)

plt.xlabel("Time")

plt.ylabel("Frequency in Hz")

plt.show()
rmse = librosa.feature.rms(x, frame_length=2, hop_length=1)[0]

print(rmse)

print(rmse.shape)

mean_rmse = np.mean(rmse)

print("RMSE:",mean_rmse)
frames = range(rmse.shape[0])

t = librosa.frames_to_time(frames)

plt.figure(figsize=(40, 10))

plt.xlabel("Time")

plt.ylabel("Magnitude")

librosa.display.waveplot(x, sr=sr, alpha=0.4)

plt.plot(t, rmse, color='r',label="Rmse")