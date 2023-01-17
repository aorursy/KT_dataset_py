import numpy as np

import matplotlib.pyplot as plt

from scipy.io.wavfile import read, write

from IPython.display import Audio

from numpy.fft import fft, ifft

from scipy.fftpack import fftfreq
# Read song and cough files

fs1, song = read("/kaggle/input/tlprojectx/song.wav")

fs2, cough1 = read("/kaggle/input/tlproject/Cough1-1.wav")

fs3, cough2 = read("/kaggle/input/tlproject/Cough2.wav")
# Select one channel for analysis

song = song[:,0]

cough1 = cough1[:,0]

cough2 = cough2[:,0]

song = song * 1
Audio(data = song, rate = fs1)
Audio(data = cough1, rate = fs1)
plt.plot(song)
plt.plot(cough1)
t1 = 10

sample_to_insert_to_1 = t1 * fs1

sample_to_insert_to_2 = sample_to_insert_to_1 + len(cough1)

song[sample_to_insert_to_1 : sample_to_insert_to_2] = song[sample_to_insert_to_1 : sample_to_insert_to_2] + cough1[:]
t1 = 50

sample_to_insert_to_1 = t1 * fs1

sample_to_insert_to_2 = sample_to_insert_to_1 + len(cough2)

song[sample_to_insert_to_1 : sample_to_insert_to_2] = song[sample_to_insert_to_1 : sample_to_insert_to_2] + cough2[:]
# Audio of song containing two coughs at 10s and 50, 51 s

Audio(data = song, rate = fs1)
# Finding index of samples where amplitude is higher than 18000 indiacting impulse signal created by cough

out1 = np.argwhere(song > 18000)

out_time = np.unique(out1 // fs1)
# Print cough detected times

print('Cough detected at', out_time[:], 's')
# Extract cough detected audio

out_audio = []

for x in out_time:

    out_audio.append(song[x*fs1:(x+2)*fs1])
Audio(data = out_audio[0], rate = fs1)
Audio(data = out_audio[1], rate = fs1)
song[sample_to_insert_to_1 : sample_to_insert_to_2] = cough2[:]
data = song[sample_to_insert_to_1 : sample_to_insert_to_2]

samples = len(data)

samplerate = fs1





datafft = fft(data)

#Get the absolute value of real and complex component:

fftabs = abs(datafft)

freqs = fftfreq(samples,1/samplerate)

plt.xlim( [10, samplerate/2] )

plt.xscale( 'log' )

plt.grid( True )

plt.xlabel( 'Frequency (Hz)' )

plt.ylabel( 'Amplitude' )

plt.title( 'Spectrum where cough is present in song')

plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])
data = cough2

samples = len(cough2)

samplerate = fs3





datafft = fft(data)

#Get the absolute value of real and complex component:

fftabs = abs(datafft)

freqs = fftfreq(samples,1/samplerate)

plt.xlim( [10, samplerate/2] )

plt.xscale( 'log' )

plt.ylabel( 'Amplitude' )

plt.grid( True )

plt.title( 'Spectrum of cough')

plt.xlabel( 'Frequency (Hz)' )

plt.plot(freqs[:int(freqs.size/2)],fftabs[:int(freqs.size/2)])