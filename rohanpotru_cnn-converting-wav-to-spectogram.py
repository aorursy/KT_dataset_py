import os

import librosa

import librosa.display

import matplotlib.pyplot as plt

import numpy as np

import json

import gzip

import binascii



import IPython.display as ipd

from scipy.io import wavfile
!wget http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

!gunzip speech_commands_v0.01.tar.gz
def is_gz_file(filepath):

    with open(filepath, 'rb') as test_f:

        return binascii.hexlify(test_f.read(2)) == b'1f8b'

    

rootDir = '.'

for dirName, subdirList, fileList in os.walk(rootDir):

    for fname in fileList:

        filepath = os.path.join(dirName,fname)

        if is_gz_file(filepath):

            f = gzip.open(filepath, 'google-speech-commands')

            json_content = json.loads(f.read())

            print(json_content)
x, sr = librosa.load('0a7c2a8d_nohash_0.wav', sr=None) #a sample that was seperately imported, it works

plt.plot(x);

plt.title('Signal');

plt.xlabel('Time (samples)');

plt.ylabel('Amplitude');

plt.show()
X = librosa.stft(x) #calculate with FFT

Xdb = librosa.amplitude_to_db(abs(X))  

plt.figure(figsize=(14, 5)) #size of the display

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz') #Sets the axes and their titles

plt.colorbar()