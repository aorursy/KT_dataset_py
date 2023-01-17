#!pip install librosa
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import librosa

ses_dosyası="/kaggle/input/ses_dosyas.wav"

x,sr=librosa.load(ses_dosyası)

print("x type:",type(x),"sr type:",type(sr))

print(x.shape,sr)
import IPython.display as ipd

ipd.Audio(ses_dosyası)
x,sr=librosa.load(ses_dosyası,sr=11025)

ipd.Audio(x,rate=sr)
print(x.shape,sr)
x, sr = librosa.load(librosa.util.example_audio_file())

librosa.output.write_wav('ses_kayit.wav', x, sr)
ses1="ses_kayit.wav"

ipd.Audio(ses1)
import matplotlib.pyplot as plt

import librosa.display

plt.figure(figsize=(10,4))

librosa.display.waveplot(x,sr=sr)
X=librosa.stft(x) #stft -> Short-time Fourier transform

Xdb=librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(18,8))

librosa.display.specshow(Xdb,sr=sr,x_axis="time",y_axis="hz")

plt.colorbar()
mfkk=librosa.feature.mfcc(x,sr=sr)

print(mfkk.shape)

mfkk
plt.figure(figsize=(15,6))

librosa.display.specshow(mfkk,x_axis="s")

plt.colorbar()
zero_crossing=librosa.zero_crossings(x)

print(sum(zero_crossing)) #Toplam sıfır geçişi sayısı
plt.plot(x[5000:5100])

plt.grid()
spec_cent=librosa.feature.spectral_centroid(x)

print(spec_cent.shape)

spec_cent
plt.figure(figsize=(10,4))

plt.semilogy(spec_cent.T)

plt.show()
plt.figure(figsize=(10,4))

spec_roll=librosa.feature.spectral_rolloff(x,sr=sr)

plt.semilogy(spec_roll.T,"r")

plt.ylabel("Hz")

print(spec_roll.shape)

spec_roll
chroma=librosa.feature.chroma_stft(x,sr=sr)

print(chroma.shape)

chroma
plt.figure(figsize=(14,6))

librosa.display.specshow(chroma,y_axis="chroma",x_axis="time")

plt.colorbar()
spec_band=librosa.feature.spectral_bandwidth(x,sr=sr)

print(spec_band,spec_band.shape)