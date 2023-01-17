import wave

import math

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.colors as colors

import os

from scipy import signal

from scipy.fft import fft, fftshift

# Loading data and pre-processing from https://www.kaggle.com/eatmygoose/cnn-detection-of-wheezes-and-crackles 

df_no_diagnosis = pd.read_csv('../input/respiratory-sound-database/demographic_info.txt', names = 

                 ['Patient number', 'Age', 'Sex' , 'Adult BMI (kg/m2)', 'Child Weight (kg)' , 'Child Height (cm)'],

                 delimiter = ' ')



diagnosis = pd.read_csv('../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/patient_diagnosis.csv', names = ['Patient number', 'Diagnosis'])



df =  df_no_diagnosis.join(diagnosis.set_index('Patient number'), on = 'Patient number', how = 'left')





root = '../input/respiratory-sound-database/Respiratory_Sound_Database/Respiratory_Sound_Database/audio_and_txt_files'

filenames = [s.split('.')[0] for s in os.listdir(path = root) if '.txt' in s]





def Extract_Annotation_Data(file_name, root):

    tokens = file_name.split('_')

    recording_info = pd.DataFrame(data = [tokens], columns = ['Patient number', 'Recording index', 'Chest location','Acquisition mode','Recording equipment'])

    recording_annotations = pd.read_csv(os.path.join(root, file_name + '.txt'), names = ['Start', 'End', 'Crackles', 'Wheezes'], delimiter= '\t')

    return (recording_info, recording_annotations)



i_list = []

rec_annotations = []

rec_annotations_dict = {}

for s in filenames:

    (i,a) = Extract_Annotation_Data(s, root)

    i_list.append(i)

    rec_annotations.append(a)

    rec_annotations_dict[s] = a

recording_info = pd.concat(i_list, axis = 0)



# Data preparation utility functions

# Used to split each individual sound file into separate sound clips containing one respiratory cycle each

# output: [filename, (sample_data:np.array, start:float, end:float, crackles:bool(float), wheezes:bool(float)) (...) ]

def get_sound_samples(recording_annotations, file_name, root, sample_rate):

    sample_data = [file_name]

    (rate, data) = read_wav_file(os.path.join(root, file_name + '.wav'), sample_rate)



    for i in range(len(recording_annotations.index)):

        row = recording_annotations.loc[i]

        start = row['Start']

        end = row['End']

        crackles = row['Crackles']

        wheezes = row['Wheezes']

        audio_chunk = slice_data(start, end, data, rate)

        sample_data.append((audio_chunk, start, end, crackles, wheezes))

    return sample_data

import scipy.io.wavfile as wf

# Resampling



# Will resample all files to the target sample rate and produce a 32bit float array



def read24bitwave(lp_wave):

    nFrames = lp_wave.getnframes()

    buf = lp_wave.readframes(nFrames)

    reshaped = np.frombuffer(buf, np.int8).reshape(nFrames, -1)

    short_output = np.empty((nFrames, 2), dtype=np.int8)

    short_output[:, :] = reshaped[:, -2:]

    short_output = short_output.view(np.int16)

    return (lp_wave.getframerate(),

            np.divide(short_output, 32768).reshape(-1))  # return numpy array to save memory via array slicing



def bitrate_channels(lp_wave):

    bps = (lp_wave.getsampwidth() / lp_wave.getnchannels())  # bytes per sample

    return (bps, lp_wave.getnchannels())



def extract2FloatArr(lp_wave, str_filename):

    (bps, channels) = bitrate_channels(lp_wave)



    if bps in [1, 2, 4]:

        (rate, data) = wf.read(str_filename)

        divisor_dict = {1: 255, 2: 32768}

        if bps in [1, 2]:

            divisor = divisor_dict[bps]

            data = np.divide(data, float(divisor))  # clamp to [0.0,1.0]

        return (rate, data)



    elif bps == 3:

        # 24bpp wave

        return read24bitwave(lp_wave)



    else:

        raise Exception('Unrecognized wave format: {} bytes per sample'.format(bps))



def resample(current_rate, data, target_rate):

    x_original = np.linspace(0, 100, len(data))

    x_resampled = np.linspace(0, 100, int(len(data) * (target_rate / current_rate)))

    resampled = np.interp(x_resampled, x_original, data)

    return (target_rate, resampled.astype(np.float32))



def read_wav_file(str_filename, target_rate):

    wav = wave.open(str_filename, mode='r')

    (sample_rate, data) = extract2FloatArr(wav, str_filename)



    if (sample_rate != target_rate):

        (_, data) = resample(sample_rate, data, target_rate)



    wav.close()

    return (target_rate, data.astype(np.float32))



def slice_data(start, end, raw_data, sample_rate):

    max_ind = len(raw_data)

    start_ind = min(int(start * sample_rate), max_ind)

    end_ind = min(int(end * sample_rate), max_ind)

    return raw_data[start_ind: end_ind]
recording_filenames = []

for i in range(len(filenames)):

    recording_filenames.append(filenames[i])

    

# Obtain lists of lung sounds according to their wheeze/crackle content for all 920 recordings

wheeze_ind = []

crackles_ind = []

norm_ind = []

both_ind = []

recording_filenames = []

for i in range(len(filenames)):

    recording_filenames.append(filenames[i])



for s in range(len(recording_filenames)):

    r = rec_annotations_dict[recording_filenames[s]]

    for i in range(len(r)):

        if r['Crackles'][i] == 1:

            if recording_filenames[s] not in crackles_ind:

                crackles_ind.append(recording_filenames[s])

        elif r['Wheezes'][i] == 1:

            if recording_filenames[s] not in wheeze_ind:

                wheeze_ind.append(recording_filenames[s])





# Generate indices for recordings with no crackles or wheezes

for i in range(len(recording_filenames)):

    if recording_filenames[i] not in crackles_ind and recording_filenames[i] not in wheeze_ind:

            norm_ind.append(recording_filenames[i])



both_ind = [x for x in wheeze_ind if x in crackles_ind]



for x in both_ind:

    wheeze_ind.remove(x)

    crackles_ind.remove(x)

print(len(crackles_ind) + len(wheeze_ind) + len(norm_ind) + len(both_ind))



# Select a recording for generating a spectrogram. Having previously generated all of the spectrograms,

# I found the recording '130_1p4_Al_mc_AKGC417L' (122 by index in the 'filenames' list) to have quite a clear wheeze feature 

str_file = '130_1p4_Al_mc_AKGC417L'

lp_test = get_sound_samples(rec_annotations_dict[str_file], str_file, root, 22000)

lp_cycles = [(d[0], d[3], d[4]) for d in lp_test[1:]]

soundclip = lp_cycles[1][0]
n_window = 550

sample_rate = 22000

(f, t, Sxx) = signal.spectrogram(soundclip, fs = 22000, nfft= n_window, nperseg=n_window)



# I used the two lines of code below to find out where to cutoff the spectrogram's frequency axis to show the 

# useful frequency range 0-2000 Hz, 

#print(f)

#print(np.where(f ==2000))



f_cut = f[0:50]

Sxx_cut = Sxx[0:50]

plt.figure(figsize=(10,5))

plt.pcolormesh(t, f_cut, Sxx_cut, cmap='nipy_spectral')

#plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin="lower")

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

n_window = 550

sample_rate = 22000

(f, t, Sxx) = signal.spectrogram(soundclip, fs = 22000, nfft= n_window, nperseg=n_window)



# I used the two lines below to find out where to cutoff the spectrogram's frequency axis to show the 

# useful frequency range 0-2000 Hz, 

#print(f)

#print(np.where(f ==2000))



f_cut = f[0:50]

Sxx_cut = Sxx[0:50]

plt.figure(figsize=(10,5))

norm = colors.LogNorm(vmin=Sxx_cut.min(), vmax=Sxx_cut.max())

plt.pcolormesh(t, f_cut, Sxx_cut, norm=norm, cmap='nipy_spectral')

#plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin="lower")

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

# Implementation of a 5th-order Butterworth filter using scipy



from scipy.signal import butter, lfilter



def butter_bandpass(lowcut, highcut, fs, order=5):

    nyq = 0.5 * fs

    low = lowcut / nyq

    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')

    return b, a





def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)

    y = lfilter(b, a, data)

    return y



# Apply the 5th order Butterworth filter to the soundclip with a lowcut of 100 Hz and a highcut of 2500 Hz

soundclip_butter = butter_bandpass_filter(soundclip, 100, 2500, sample_rate)
(f, t, Sxx) = signal.spectrogram(soundclip_butter, fs = 22000, nfft= n_window, nperseg=n_window)

f_cut = f[0:50]

Sxx_cut = Sxx[0:50]

plt.figure(figsize=(10,5))

norm = colors.LogNorm(vmin=Sxx_cut.min(), vmax=Sxx_cut.max())

plt.pcolormesh(t, f_cut, Sxx_cut, norm=norm, cmap='nipy_spectral')

#plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin="lower")

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')

(f, t, Sxx) = signal.spectrogram(soundclip_butter, fs = 22000, nfft= n_window, nperseg=n_window)

f_cut = f[0:50]

Sxx_cut = Sxx[0:50]

plt.figure(figsize=(10,5))

norm = colors.LogNorm(vmin=Sxx_cut.min(), vmax=Sxx_cut.max())

plt.pcolormesh(t, f_cut, Sxx_cut, norm=norm, cmap='rainbow')

#plt.imshow(a, aspect='auto', cmap=plt.get_cmap(m), origin="lower")

plt.ylabel('Frequency [Hz]')

plt.xlabel('Time [sec]')
# Code to output all spectrograms in the list containing recordings with wheezes.



for f in range(len(wheeze_ind)):

    str_file = wheeze_ind[f]

    lp_test = get_sound_samples(rec_annotations_dict[str_file], str_file, root, 22000)

    lp_cycles = [(d[0], d[3], d[4]) for d in lp_test[1:]]

    soundclip = lp_cycles[1][0]

    # Add Butterworth filter

    soundclip_butter = butter_bandpass_filter(soundclip, 100, 2500, sample_rate)

    (f, t, Sxx) = signal.spectrogram(soundclip_butter, fs=22000, nfft=n_window, nperseg=n_window)

    f_cut = f[0:50]

    Sxx_cut = Sxx[0:50]

    norm = colors.LogNorm(vmin=1e-015, vmax=Sxx_cut.max())

    plt.figure(figsize=(10, 5))

    plt.pcolormesh(t, f_cut, Sxx_cut, norm=norm, cmap='nipy_spectral')

    plt.ylabel('Frequency [Hz]')

    plt.xlabel('Time [sec]')

    plt.savefig('../LungSounds/Plots/spec_wheeze_butter_{}.png'.format(str_file))

    plt.close()