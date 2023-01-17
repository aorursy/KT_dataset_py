DATA_DIR = 'data'

!mkdir {DATA_DIR} -p
# Only on Linux, Mac and Windows WSL

!wget http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS.tgz

!tar -C {DATA_DIR} -zxf ST-AEDS-20180100_1-OS.tgz

!rm ST-AEDS-20180100_1-OS.tgz
import os

from IPython.display import Audio



audio_files = os.listdir(DATA_DIR)

len(audio_files), audio_files[:10]
example = DATA_DIR + "/" + audio_files[0]

Audio(example)
Audio(DATA_DIR + "/" + audio_files[1])
Audio(DATA_DIR + "/" + audio_files[823])
import librosa
y, sr = librosa.load(example, sr=None)
print("Sample rate  :", sr)

print("Signal Length:", len(y))

print("Duration     :", len(y)/sr, "seconds")
print("Type  :", type(y))

print("Signal: ", y)

print("Shape :", y.shape)
Audio(y, rate=sr)
Audio(y, rate=sr/2)
Audio(y, rate=sr*2)
y_new, sr_new = librosa.load(example, sr=sr*2)

Audio(y_new, rate=sr_new)
y_new, sr_new = librosa.load(example, sr=sr/2)

Audio(y_new, rate=sr_new)
%matplotlib inline

import matplotlib.pyplot as plt

import librosa.display
plt.figure(figsize=(15, 5))

librosa.display.waveplot(y, sr=sr);
import numpy as np
# Adapted from https://musicinformationretrieval.com/audio_representation.html

# An amazing open-source resource, especially if music is your sub-domain.

def make_tone(freq, clip_length=1, sr=16000):

    t = np.linspace(0, clip_length, int(clip_length*sr), endpoint=False)

    return 0.1*np.sin(2*np.pi*freq*t)

clip_500hz = make_tone(500)

clip_5000hz = make_tone(5000)
Audio(clip_500hz, rate=sr)
Audio(clip_5000hz, rate=sr)
plt.figure(figsize=(15, 5))

plt.plot(clip_500hz[0:64]);
plt.figure(figsize=(15, 5))

plt.plot(clip_5000hz[0:64]);
plt.figure(figsize=(15, 5))

plt.plot((clip_500hz + clip_5000hz)[0:64]);
Audio(clip_500hz + clip_5000hz, rate=sr)
clip_500_to_1000 = np.concatenate([make_tone(500), make_tone(1000)])

clip_5000_to_5500 = np.concatenate([make_tone(5000), make_tone(5500)])
# first half of the clip is 500hz, 2nd is 1000hz

Audio(clip_500_to_1000, rate=sr)
# first half of the clip is 5000hz, 2nd is 5500hz

Audio(clip_5000_to_5500, rate=sr)
sg0 = librosa.stft(y)

sg_mag, sg_phase = librosa.magphase(sg0)

librosa.display.specshow(sg_mag);
sg1 = librosa.feature.melspectrogram(S=sg_mag, sr=sr)

librosa.display.specshow(sg1);
sg2 = librosa.amplitude_to_db(sg1, ref=np.min)

librosa.display.specshow(sg2, sr=16000, y_axis='mel', fmax=8000, x_axis='time')

plt.colorbar(format='%+2.0f dB')

plt.title('Mel spectrogram');
sg2.min(), sg2.max(), sg2.mean()
type(sg2), sg2.shape
plt.imshow(sg2);
# Clean up

!rm -rf data