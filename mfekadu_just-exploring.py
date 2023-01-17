!yes | apt-get install libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0

!yes | apt-get install ffmpeg libav-tools

!yes | apt-get install python-gnuradio-audio-portaudio

!yes | pip install --upgrade pip

!yes | pip install pyaudio
import pyaudio # for reading & writing audio files

import struct # for unpacking chunks of audio for plotting

import IPython.display as ipd  # To play sound in the notebook

import wave # opening .wav files

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualizations

%matplotlib inline

import os # operating system



from os.path import join





print(os.listdir("../input"))



DATA_DIR = "../input/data"

FIRST_TRAIN_SAMPLE_DIR = join(DATA_DIR, "TRAIN", "DR1", "FCJF0")

wav_files = [n for n in os.listdir(FIRST_TRAIN_SAMPLE_DIR) if n.endswith("wav")]

print(wav_files)
ONE_BYTE = 1024

CHUNK = ONE_BYTE * 2 # chunk is how many audio samples to process at a time

FORMAT = pyaudio.paInt16 # 16 bit int sample size

CHANNELS = 1 # mono audio

RATE = 16000 # KHz # samples per second

MODEL_INPUT_SECONDS = 1.5
# open a .wav file

filepath = join(FIRST_TRAIN_SAMPLE_DIR, wav_files[0])

wf = wave.open(filepath, 'rb')
# read data

data = wf.readframes(nframes=CHUNK)

# convert data to integers

data_ints = struct.unpack(str(2*CHUNK) + 'B', data)

# convert to numpy array 8 bit ints between 

data_ints = np.array(data_ints, dtype='int8') + 127
data_ints
print("wf.getsampwidth(): ", wf.getsampwidth(), "...what is this???")

print("wf.getnchannels(): ", wf.getnchannels(), "mono" if wf.getnchannels()==1 else "stereo")

print("wf.getframerate(): ", wf.getframerate(), "KHz")

assert(wf.getframerate() == RATE) # test case

print("wf.getnframes(): ", wf.getnframes(), "...should this match the end timestamp in .WRD?")

print('len_data=',len(data), 'CHUNK=',CHUNK)

print("str(2*CHUNK)+'B'=",str(2*CHUNK) + 'B')
fig, axis = plt.subplots()

axis.plot(data_ints, '-')
print(wav_files)
print(wav_files[0])

fpath = join(FIRST_TRAIN_SAMPLE_DIR, wav_files[0])

ipd.Audio(fpath)
print(wav_files[1])

fpath = join(FIRST_TRAIN_SAMPLE_DIR, wav_files[1])

ipd.Audio(fpath)
print(wav_files[2])

fpath = join(FIRST_TRAIN_SAMPLE_DIR, wav_files[2])

ipd.Audio(fpath)
print(wav_files[3])

fpath = join(FIRST_TRAIN_SAMPLE_DIR, wav_files[3])

ipd.Audio(fpath)
wrd_file = wav_files[4].replace('.WAV.wav', '') + '.WRD'

wrd_path = join(FIRST_TRAIN_SAMPLE_DIR, wrd_file)

fpath = join(FIRST_TRAIN_SAMPLE_DIR, wav_files[4])

print(wrd_path)

print(fpath)



# print the .WRD file contents

wrd_file = open(wrd_path)

print()

print(wrd_file.read())

wrd_file.close()



# output a GUI to playback audio

ipd.Audio(fpath)
wav = wave.open(fpath)

print("Sampling (frame) rate = ", wav.getframerate())

print("Total samples (frames) = ", wav.getnframes())

print("Duration = ", wav.getnframes()/wav.getframerate())

wav.close()
# Using scipy

from scipy.io import wavfile

rate, data = wavfile.read(fpath)

print("Sampling (frame) rate = ", rate)

print("Total samples (frames) = ", data.shape)
plt.plot(data, '-', );
wrd_file = wav_files[4].replace('.WAV.wav', '') + '.WRD'

wrd_path = join(FIRST_TRAIN_SAMPLE_DIR, wrd_file)

fpath = join(FIRST_TRAIN_SAMPLE_DIR, wav_files[4])

print(wrd_path)

print(fpath)



# print the .WRD file contents

wrd_file = open(wrd_path)

print()

print(wrd_file.read())

wrd_file.close()



# output a GUI to playback audio

ipd.Audio(fpath)
plt.plot(data, '-', );

font = {'size':24, 'weight':'bold', 'rotation':90, 'color':'w','backgroundcolor':'k'}

plt.text(2260, 0, 'critical',fontdict=font)

plt.text(7895, 1000, 'equipment',fontdict=font)

plt.text(13781, 0, 'needs',fontdict=font)

plt.text(18101, 0, 'proper',fontdict=font)

plt.text(23628, 2000, 'maintenance',fontdict=font)

plt.show()
plt.plot(data, '-', );

font = {'size':24, 'weight':'bold', 'rotation':90}

plt.text(2260, 0, 'critical',fontdict=font)

plt.text(7895, 1000, 'equipment',fontdict=font)

plt.text(13781, 0, 'needs',fontdict=font)

plt.text(18101, 0, 'proper',fontdict=font)

plt.text(23628, 2000, 'maintenance',fontdict=font)

plt.show()
plt.plot(data, '-', );

plt.show()