!pip install nlpaug
import pandas as pd

import numpy as np

import librosa

import librosa.display as librosa_display

import IPython.display as ipd

import nlpaug

import nlpaug.augmenter.audio as naa

import matplotlib.pyplot as plt
perfal = '/kaggle/input/birdsong-recognition/train_audio/perfal/XC463087.mp3'

lotduc = '/kaggle/input/birdsong-recognition/train_audio/lotduc/XC121426.mp3' 

rewbla = '/kaggle/input/birdsong-recognition/train_audio/rewbla/XC135672.mp3' 

warvir = '/kaggle/input/birdsong-recognition/train_audio/warvir/XC192521.mp3' 

lecthr = '/kaggle/input/birdsong-recognition/train_audio/lecthr/XC141435.mp3' 
import IPython.display as ipd

data, sr = librosa.load(perfal)

ipd.Audio(data, rate=sr)

librosa.display.waveplot(data.astype('float'), sr=sr,x_axis=None)

plt.title('original')
def pitch_speed(filename):

    data, sr = librosa.load(filename)

    pitch_speed = data.copy()

    length_change = np.random.uniform(low=0.8, high = 1)

    speed_fac = 1.0  / length_change

    print("resample length_change = ",length_change)

    tmp = np.interp(np.arange(0,len(pitch_speed),speed_fac),np.arange(0,len(pitch_speed)),pitch_speed)

    minlen = min(pitch_speed.shape[0], tmp.shape[0])

    pitch_speed *= 0

    pitch_speed[0:minlen] = tmp[0:minlen]

    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(pitch_speed, sr=sr, color='r', alpha=0.25)

    plt.title('augmented pitch and speed')

    return ipd.Audio(data, rate=sr)
pitch_speed(perfal)
pitch_speed(lotduc)
def pitch(filename):

    data, sr = librosa.load(filename)

    y_pitch = data.copy()

    bins_per_octave = 12

    pitch_pm = 2

    pitch_change =  pitch_pm * 2*(np.random.uniform())   

    print("pitch_change = ",pitch_change)

    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'), 

                                          sr, n_steps=pitch_change, 

                                          bins_per_octave=bins_per_octave)

    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(y_pitch, sr=sr, color='r', alpha=0.25)

    plt.title('augmented pitch only')

    plt.tight_layout()

    plt.show()

    return ipd.Audio(data, rate=sr)
pitch(perfal)
pitch(lotduc)
def speed(filename):

    data, sr = librosa.load(filename)

#     speed_change = np.random.uniform(low=0.9,high=1.1)

#     print("speed_change = ",speed_change)

#     tmp = librosa.effects.time_stretch(data.astype('float64'), speed_change)

#     minlen = min(data.shape[0], tmp.shape[0])

#     data *= 0 

#     data[0:minlen] = tmp[0:minlen]

#     librosa.display.waveplot(data.astype('float'), sr=sr,x_axis=None)

    aug = naa.SpeedAug()

    augmented_data = aug.augment(data)



    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)

    plt.title('augmented speed only')

    plt.tight_layout()

    plt.show()

    return ipd.Audio(augmented_data, rate=sr)
speed(perfal)
speed(lotduc)
def augmentation(filename):

    data, sr = librosa.load(filename)

    y_aug = data.copy()

    dyn_change = np.random.uniform(low=1.5,high=3)

    print("dyn_change = ",dyn_change)

    y_aug = y_aug * dyn_change

    print(y_aug[:50])

    print(data[:50])

    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(y_aug, sr=sr, color='r', alpha=0.25)

    plt.title('amplify value')

    return ipd.Audio(y_aug, rate=sr)
augmentation(perfal)
augmentation(lotduc)
def add_noise(filename):

    data, sr = librosa.load(filename)

    y_noise = data.copy()

    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html

    noise_amp = 0.005*np.random.uniform()*np.amax(y_noise)

    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])

    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(y_noise, sr=sr, color='r', alpha=0.25)

    return ipd.Audio(y_noise, rate=sr)
add_noise(perfal)
add_noise(lotduc)
def random_shift(filename):

    data, sr = librosa.load(filename)

    y_shift = data.copy()

    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length

    print("timeshift_fac = ",timeshift_fac)

    start = int(y_shift.shape[0] * timeshift_fac)

    print(start)

    if (start > 0):

        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]

    else:

        y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]

    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(y_shift, sr=sr, color='r', alpha=0.25)

    return ipd.Audio(y_shift, rate=sr)
random_shift(perfal)
random_shift(lotduc)
def hpss(filename):

    data, sr = librosa.load(filename)

    y_hpss = librosa.effects.hpss(data.astype('float64'))

    print(y_hpss[1][:10])

    print(data[:10])

    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(y_hpss[1], sr=sr, color='r', alpha=0.25)

    plt.title('apply hpss')

    return ipd.Audio(y_hpss[1], rate=sr)
hpss(perfal)
hpss(lotduc)
def streching(filename):

    data, sr = librosa.load(filename)

    input_length = len(data)

    streching = data.copy()

    streching = librosa.effects.time_stretch(streching.astype('float'), 1.1)

    if len(streching) > input_length:

        streching = streching[:input_length]

    else:

        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")

    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(streching, sr=sr, color='r', alpha=0.25)

    

    plt.title('stretching')

    return ipd.Audio(streching, rate=sr)
streching(perfal)
streching(lotduc)
def crop(filename):

    data, sr = librosa.load(filename)

    aug = naa.CropAug(sampling_rate=sr)

    augmented_data = aug.augment(data)



    librosa_display.waveplot(augmented_data, sr=sr, alpha=0.5)

    librosa_display.waveplot(data, sr=sr, color='r', alpha=0.25)



    plt.tight_layout()

    plt.show()



    return ipd.Audio(augmented_data, rate=sr)
crop(perfal)
crop(lotduc)
def loudnessaug(filename):

    data, sr = librosa.load(filename)

    aug = naa.LoudnessAug(loudness_factor=(2, 5))

    augmented_data = aug.augment(data)



    librosa_display.waveplot(augmented_data, sr=sr, alpha=0.25)

    librosa_display.waveplot(data, sr=sr, color='r', alpha=0.5)



    plt.tight_layout()

    plt.show()



    return ipd.Audio(augmented_data,rate=sr)
loudnessaug(perfal)
loudnessaug(lotduc)
def mask(filename):

    data, sr = librosa.load(filename)

    aug = naa.MaskAug(sampling_rate=sr, mask_with_noise=False)

    augmented_data = aug.augment(data)



    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)



    plt.tight_layout()

    plt.show()

    

    return ipd.Audio(augmented_data, rate=sr)
mask(perfal)
mask(lotduc)
def shift(filename):

    data, sr = librosa.load(filename)

    aug = naa.ShiftAug(sampling_rate=sr)

    augmented_data = aug.augment(data)



    librosa_display.waveplot(data, sr=sr, alpha=0.5)

    librosa_display.waveplot(augmented_data, sr=sr, color='r', alpha=0.25)



    plt.tight_layout()

    plt.show()

    

    return ipd.Audio(augmented_data, rate=sr)
shift(perfal)
shift(lotduc)