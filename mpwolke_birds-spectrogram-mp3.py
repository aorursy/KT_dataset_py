# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

songs = []

cwd = '/kaggle/input/birdsongs-from-europe/mp3/'



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        songs.append(filename)

data = pd.read_csv('/kaggle/input/birdsongs-from-europe/metadata.csv')

songs.pop(0)



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print(data.head())

print(data.info(verbose=True))
! pip install pydub
from pydub import AudioSegment

import IPython



# We will listen to this file:

# 213_1p5_Pr_mc_AKGC417L.wav

file = '/kaggle/input/birdsongs-from-europe/mp3/Aegolius-funereus-131493.mp3'

print(cwd+songs[0])

IPython.display.Audio(cwd+songs[2])
# https://www.kaggle.com/rakibilly/extract-audio-starter

import subprocess

import glob

import os

from pathlib import Path

import shutil

from zipfile import ZipFile
! tar xvf ../input/ffmpeg-static-build/ffmpeg-git-amd64-static.tar.xz
# Convert MP3s to WAV for easy conversion to numpy arrays:

output_format = 'wav'  # can also use aac, wav, etc

output_dir = Path(f"{output_format}s")

Path(output_dir).mkdir(exist_ok=True, parents=True)



#Only do first 50 because notebook memory limitations...

for song in songs[:50]:

    file = cwd+song

    file_name = song.replace(".mp3","")

    command = f"../working/ffmpeg-git-20191209-amd64-static/ffmpeg -i {file} -ab 192000 -ac 2 -ar 44100 -vn {output_dir/file_name}.{output_format}"

    subprocess.call(command, shell=True)
from scipy.io.wavfile import read, write

#a = read("adios.wav")

wavs = []

np_arrays = []

for dirname, _, filenames in os.walk('/kaggle/working/wavs/'):

    for filename in filenames:

        wav_file = dirname+filename

        #print(wav_file)

        wavs.append(wav_file)

        try:

            fs, io_file = read(wav_file)

        except ValueError:

            continue

        data = np.array(io_file,dtype=float)

        wav_info= {

            'name': filename,

            'fs' : fs,

            'left': data[:,0],

            'right': data[:,1]

        }

        

        np_arrays.append(wav_info)



print("Succesfully converted: "+str(len(np_arrays)))
from scipy import signal

from scipy.fft import fftshift

import matplotlib.pyplot as plt



song_data = np_arrays[26]

start = 0

end = 10



if end != None:

    wav = song_data['left'][fs*start:fs*end]

else:

    wav = song_data['left'][fs*start:]

fs = song_data['fs']

plt.specgram(wav,Fs=fs)

plt.ylim(top=15000)

print(song_data['name'].replace(".wav",""))

plt.show() 



IPython.display.Audio(wav, rate=fs)
! pip install pyyawt
# Load a noisy signal

# Phylloscopus-collybita-171141



song_data = np_arrays[4]

start = 1

end = 12



if end != None:

    wav = song_data['left'][fs*start:fs*end]

else:

    wav = song_data['left'][fs*start:]

fs = song_data['fs']

plt.specgram(wav,Fs=fs)

plt.ylim(top=15000)

print(song_data['name'].replace(".wav",""))

plt.show() 



IPython.display.Audio(wav, rate=fs)
import seaborn as sns

import pywt

import pyyawt



stds = []

means = []

decomps = []

thrs = []

wavelets = pywt.wavedec(wav, 'db5', level=10)



for i, wavelet in enumerate(wavelets):

    thrs.append(pyyawt.thselect(wavelet, 'heursure'))

    stds.append(wavelet.std(0))

    means.append(wavelet.mean(0))

    decomps.append(wavelet)

    

    #ax[i+1,0].plot(wavelet)

    #ax[i+1,0].plot(wavelet)

    #sns.distplot(wavelet, ax=ax[i+1,1], hist=False, vertical=True)



thresholded = []



fig, ax = plt.subplots(len(wavelets), figsize=(20,20))





for i, decomp in enumerate(decomps):

    thresh =((np.amax(decomp)-means[i])*thrs[i])

    print(thrs[i], np.amax(decomp), thresh)

    thresholded.append(pywt.threshold(decomp, thresh, 'soft'))

    ax[i].plot(wavelets[i])

    ax[i].plot(thresholded[i])



print("Denoised: "+song_data['name'].replace(".wav",""))

reconstructed = pywt.waverec(thresholded, 'db5')

plt.specgram(reconstructed,Fs=fs)

plt.show()

IPython.display.Audio(reconstructed, rate=fs)