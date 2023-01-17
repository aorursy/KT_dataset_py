import numpy as np

import librosa

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

import os

import scipy

import librosa.display

from IPython.display import Audio

os.system('tar -zxf ../input/train.7z')
zero = '../input/train/audio/zero/'

random_zero = zero+os.listdir(zero)[0]

Audio(random_zero)
sample_rate, samples = scipy.io.wavfile.read(random_zero)

plt.figure(figsize=(12, 4))

librosa.display.waveplot(samples.astype('float'), sr=sample_rate)

plt.show()
X = librosa.stft(samples.astype('float'))

Xdb = librosa.amplitude_to_db(X)

plt.figure(figsize=(12, 5))

librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')

plt.show()
y_pitch_speed = samples.copy()

# you can change low and high here

length_change = np.random.uniform(low=0.8, high = 1)

speed_fac = 1.0  / length_change

print("resample length_change = ",length_change)

tmp = np.interp(np.arange(0,len(y_pitch_speed),speed_fac),np.arange(0,len(y_pitch_speed)),y_pitch_speed)

minlen = min(y_pitch_speed.shape[0], tmp.shape[0])

y_pitch_speed *= 0

y_pitch_speed[0:minlen] = tmp[0:minlen]

Audio(y_pitch_speed, rate=sample_rate)
y_pitch = samples.copy()

bins_per_octave = 12

pitch_pm = 2

pitch_change =  pitch_pm * 2*(np.random.uniform())   

print("pitch_change = ",pitch_change)

y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), 

                                      sample_rate, n_steps=pitch_change, 

                                      bins_per_octave=bins_per_octave)

Audio(y_pitch, rate=sample_rate)
y_speed = samples.copy()

speed_change = np.random.uniform(low=0.9,high=1.1)

print("speed_change = ",speed_change)

tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)

minlen = min(y_speed.shape[0], tmp.shape[0])

y_speed *= 0 

y_speed[0:minlen] = tmp[0:minlen]

Audio(y_speed, rate=sample_rate)
y_aug = samples.copy()

dyn_change = np.random.uniform(low=1.5,high=3)

print("dyn_change = ",dyn_change)

y_aug = y_aug * dyn_change

print(y_aug[:50])

print(samples[:50])

Audio(y_aug, rate=sample_rate)
y_noise = samples.copy()

# you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html

noise_amp = 0.005*np.random.uniform()*np.amax(y_noise)

y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])

Audio(y_noise, rate=sample_rate)
y_shift = samples.copy()

timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length

print("timeshift_fac = ",timeshift_fac)

start = int(y_shift.shape[0] * timeshift_fac)

print(start)

if (start > 0):

    y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]

else:

    y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]

Audio(y_shift, rate=sample_rate)
y_hpss = librosa.effects.hpss(samples.astype('float64'))

print(y_hpss[1][:10])

print(samples[:10])

Audio(y_hpss[1], rate=sample_rate)
samples[(samples > 200) | (samples < -200)]
sampling=samples[(samples > 200) | (samples < -200)]

shifted_silent =sampling.tolist()+np.zeros((samples.shape[0]-sampling.shape[0])).tolist()

Audio(shifted_silent, rate=sample_rate)
input_length = len(samples)

streching = samples.copy()

streching = librosa.effects.time_stretch(streching.astype('float'), 1.1)

if len(streching) > input_length:

    streching = streching[:input_length]

else:

    streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")

Audio(streching, rate=sample_rate)
plt.figure(figsize=(12, 4*9))

plt.subplot(9, 1, 1)

librosa.display.waveplot(samples.astype('float'), sr=sample_rate,x_axis=None)

plt.title('original')

plt.subplot(9, 1, 2)

librosa.display.waveplot(y_pitch_speed.astype('float'), sr=sample_rate,x_axis=None)

plt.title('augmented pitch and speed')

plt.subplot(9, 1, 3)

librosa.display.waveplot(y_pitch, sr=sample_rate,x_axis=None)

plt.title('augmented pitch only')

plt.subplot(9, 1, 4)

librosa.display.waveplot(y_speed.astype('float'), sr=sample_rate,x_axis=None)

plt.title('augmented speed only')

plt.subplot(9, 1, 5)

librosa.display.waveplot(y_aug.astype('float'), sr=sample_rate,x_axis=None)

plt.title('amplify value')

plt.subplot(9, 1, 6)

librosa.display.waveplot(y_noise, sr=sample_rate,x_axis=None)

plt.title('augmented noise')

plt.subplot(9, 1, 7)

librosa.display.waveplot(y_hpss[1], sr=sample_rate,x_axis=None)

plt.title('apply hpss')

plt.subplot(9, 1, 8)

librosa.display.waveplot(np.array(shifted_silent).astype('float'), sr=sample_rate,x_axis=None)

plt.title('shift silent to right')

plt.subplot(9, 1, 9)

librosa.display.waveplot(streching, sr=sample_rate)

plt.title('stretching')

plt.show()