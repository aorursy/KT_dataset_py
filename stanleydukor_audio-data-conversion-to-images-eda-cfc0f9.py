import matplotlib.pyplot as plt

from matplotlib.backend_bases import RendererBase

from scipy import signal

from scipy.io import wavfile

#import soundfile as sf

import os

import numpy as np

from PIL import Image

from scipy.fftpack import fft



%matplotlib inline
audio_path = '../input/train/audio/'

pict_Path = '../input/picts/train/'

test_pict_Path = '../input/picts/test/'

test_audio_path = '../input/test/audio/'

samples = []
subFolderList = []

for x in os.listdir(audio_path):

    if os.path.isdir(audio_path + '/' + x):

        subFolderList.append(x)
sample_audio = []

total = 0

for x in subFolderList:

    

    # get all the wave files

    all_files = [y for y in os.listdir(audio_path + x) if '.wav' in y]

    total += len(all_files)

    # collect the first file from each dir

    sample_audio.append(audio_path  + x + '/'+ all_files[0])

    

    # show file counts

    print('count: %d : %s' % (len(all_files), x ))

print(total)
sample_audio[0]
def log_specgram(audio, sample_rate, window_size=20,

                 step_size=10, eps=1e-10):

    nperseg = int(round(window_size * sample_rate / 1e3))

    noverlap = int(round(step_size * sample_rate / 1e3))

    freqs, _, spec = signal.spectrogram(audio,

                                    fs=sample_rate,

                                    window='hann',

                                    nperseg=nperseg,

                                    noverlap=noverlap,

                                    detrend=False)

    return freqs, np.log(spec.T.astype(np.float32) + eps)
fig = plt.figure(figsize=(10,10))



# for each of the samples

for i, filepath in enumerate(sample_audio[:9]):

    # Make subplots

    plt.subplot(3,3,i+1)

    

    # pull the labels

    label = filepath.split('/')[-2]

    plt.title(label)

    

    # create spectogram

    samplerate, test_sound  = wavfile.read(filepath)

    _, spectrogram = log_specgram(test_sound, samplerate)

    

    plt.imshow(spectrogram.T, aspect='auto', origin='lower')

    plt.axis('off')
five_samples = [audio_path + 'five/' + y for y in os.listdir(audio_path + 'five/')[:6]]



fig = plt.figure(figsize=(10,10))



for i, filepath in enumerate(five_samples):

    # Make subplots

    plt.subplot(3,3,i+1)

    

    # pull the labels

    label = filepath.split('/')[-1]

    plt.title('"five": '+label)

    

    # create spectogram

    # create spectogram

    samplerate, test_sound  = wavfile.read(filepath)

    _, spectrogram = log_specgram(test_sound, samplerate)

    

    plt.imshow(spectrogram.T, aspect='auto', origin='lower')

    plt.axis('off')
fig = plt.figure(figsize=(8,20))

for i, filepath in enumerate(sample_audio[:6]):

    plt.subplot(9,1,i+1)

    samplerate, test_sound  = wavfile.read(filepath)

    plt.title(filepath.split('/')[-2])

    plt.axis('off')

    plt.plot(test_sound)
fig = plt.figure(figsize=(8,20))

for i, filepath in enumerate(five_samples):

    plt.subplot(9,1,i+1)

    samplerate, test_sound = wavfile.read(filepath)

    plt.title(filepath.split('/')[-2])

    plt.axis('off')

    plt.plot(test_sound)
def wav2img(wav_path, targetdir='', figsize=(4,4)):

    """

    takes in wave file path

    and the fig size. Default 4,4 will make images 288 x 288

    """



    fig = plt.figure(figsize=figsize)    

    # use soundfile library to read in the wave files

    samplerate, test_sound  = wavfile.read(filepath)

    _, spectrogram = log_specgram(test_sound, samplerate)

    

    ## create output path

    output_file = wav_path.split('/')[-1].split('.wav')[0]

    output_file = targetdir +'/'+ output_file

    #plt.imshow(spectrogram.T, aspect='auto', origin='lower')

    plt.imsave('%s.png' % output_file, spectrogram)

    plt.close()
def wav2img_waveform(wav_path, targetdir='', figsize=(4,4)):

    samplerate,test_sound  = wavfile.read(sample_audio[0])

    fig = plt.figure(figsize=figsize)

    plt.plot(test_sound)

    plt.axis('off')

    output_file = wav_path.split('/')[-1].split('.wav')[0]

    output_file = targetdir +'/'+ output_file

    plt.savefig('%s.png' % output_file)

    plt.close()