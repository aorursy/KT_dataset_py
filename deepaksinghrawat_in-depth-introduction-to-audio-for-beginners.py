! git clone https://github.com/mogwai/fastai_audio.git
import sys

sys.path.append('..')

from fastai.vision import *

from fastai_audio.audio import *

import matplotlib.pyplot as plt

import math
data_url = 'http://www.openslr.org/resources/45/ST-AEDS-20180100_1-OS'

data_folder = datapath4file(url2name(data_url))

if not os.path.exists(data_folder): untar_data(data_url, dest=data_folder)
from IPython.display import Audio

audio_files = data_folder.ls()

example = audio_files[0]

Audio(str(example))
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
import librosa.display

plt.figure(figsize=(15, 5))

librosa.display.waveplot(y, sr=sr)
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

plt.plot(clip_500hz[0:64])
plt.figure(figsize=(15, 5))

plt.plot(clip_5000hz[0:64])
clip_500_to_1000 = np.concatenate([make_tone(500), make_tone(1000)])

clip_5000_to_5500 = np.concatenate([make_tone(5000), make_tone(5500)])
# first half of the clip is 500hz, 2nd is 1000hz

Audio(clip_500_to_1000, rate=sr)
# first half of the clip is 5000hz, 2nd is 5500hz

Audio(clip_5000_to_5500, rate=sr)
sg0 = librosa.stft(y)

sg_mag, sg_phase = librosa.magphase(sg0)

display(librosa.display.specshow(sg_mag))
sg1 = librosa.feature.melspectrogram(S=sg_mag, sr=sr)

display(librosa.display.specshow(sg1))
sg2 = librosa.amplitude_to_db(sg1, ref=np.min)

librosa.display.specshow(sg2)
# code adapted from the librosa.feature.melspectrogram documentation

librosa.display.specshow(sg2, sr=16000, y_axis='mel', fmax=8000, x_axis='time')

plt.colorbar(format='%+2.0f dB')

plt.title('Mel spectrogram')
sg2.min(), sg2.max(), sg2.mean()
print(type(sg2))

sg2.shape
Image.show(torch.from_numpy(sg2).unsqueeze(0), figsize=(15, 5), cmap=None)
y, sr = librosa.load(example)

display(Audio(y, rate=sr))

sg = librosa.feature.melspectrogram(y, sr=16000,  n_fft=2048, hop_length=512, power=1.0, n_mels=128, fmin=0.0, fmax=None)

db_spec = librosa.amplitude_to_db(sg, ref=1.0, amin=1e-05, top_db=80.0)

librosa.display.specshow(db_spec, y_axis='mel', fmax=8000, x_axis='time')
Image(torch.from_numpy(db_spec).unsqueeze(0))
# Adapted from https://musicinformationretrieval.com/audio_representation.html

# An amazing open-source resource, especially if music is your sub-domain.

def make_tone(freq, clip_length=1, sr=16000):

    t = np.linspace(0, clip_length, int(clip_length*sr), endpoint=False)

    return 0.1*np.sin(2*np.pi*freq*t)
def add_3_random_tones(clip_length=1, sr=16000):

    tone_list = []

    for i in range(3):

        frequency = random.randint(500,8000)

        tone_list.append(make_tone(frequency, clip_length, sr))

        print(f"Frequency {i+1}: {frequency}")

    return sum(tone_list)
sr = 16000

signal = add_3_random_tones(sr=sr)
display(Audio(signal, rate=sr))

plt.figure(figsize=(15, 5))

plt.plot(signal[200:400])
# Code adapted from https://musicinformationretrieval.com/fourier_transform.html and the original

# implementation of fastai audio by John Hartquist at https://github.com/sevenfx/fastai_audio/

def fft_and_display(signal, sr):

    ft = scipy.fftpack.fft(signal, n=len(signal))

    ft = ft[:len(signal)//2+1]

    ft_mag = np.absolute(ft)

    f = np.linspace(0, sr/2, len(ft_mag)) # frequency variable

    plt.figure(figsize=(13, 5))

    plt.plot(f, ft_mag) # magnitude spectrum

    plt.xlabel('Frequency (Hz)')

    

fft_and_display(signal, sr)
for i in range(5):

    signal += add_3_random_tones(sr=sr)
fft_and_display(signal, sr)
y, sr = librosa.load(example, sr=16000)

fft_and_display(y, sr)
s1 = add_3_random_tones(sr=sr)

s2 = add_3_random_tones(sr=sr)

s1_plus_s2 = np.add(s1, s2)

s1_then_s2 = np.concatenate([s1, s2])

display(Audio(s1_plus_s2, rate=sr))

display(Audio(s1_then_s2, rate=sr))
fft_and_display(s1_plus_s2, sr)
fft_and_display(s1_then_s2, sr)
def stft_and_display(signal, n_fft=512, hop_length=128, to_db_scale=False, n_mels=128, mel_scale=False, 

                     top_db=80, show_shape=False):

    stft = librosa.stft(signal, n_fft, hop_length)

    real_portion = abs(stft)

    if(mel_scale):   real_portion = librosa.feature.melspectrogram(S=real_portion, n_fft=n_fft, n_mels=n_mels)

    if(to_db_scale): real_portion = librosa.amplitude_to_db(real_portion, top_db)

    if(show_shape):  print("Shape: {}x{}".format(*real_portion.shape))

    display(Image(torch.from_numpy(real_portion).unsqueeze(0)))

display(Audio(s1_plus_s2, rate=sr))    

stft_and_display(s1_plus_s2)

display(Audio(s1_then_s2, rate=sr))    

stft_and_display(s1_then_s2)
for n_fft in range(100, 2100, 500):

    print("n_fft =", n_fft)

    stft_and_display(s1_then_s2, n_fft=n_fft)
y, sr = librosa.load(example, sr=16000)

for n_fft in range(50, 1050, 200):

    print("n_fft =", n_fft)

    stft_and_display(y, n_fft=n_fft)
for n_fft in range(50, 1050, 200):

    print("n_fft =", n_fft)

    stft_and_display(y, n_fft=n_fft, mel_scale=True, to_db_scale=True)
for hop_length in range(50, 550, 100):

    print("hop_length =", hop_length )

    stft_and_display(y, n_fft=850, hop_length=hop_length, mel_scale=True, to_db_scale=True)
for hop_length in range(100, 500, 100):

    print("Sig length   :", len(y))

    print("hop_length   :", hop_length)

    print("SigLen/HopLen:", len(y)/hop_length)

    print("Floor + 1    :", int(len(y)/hop_length)+1)

    stft_and_display(y, n_fft=850, hop_length=hop_length, mel_scale=True, to_db_scale=True, show_shape=True)
for n_mels in range(50, 1050, 250):

    print("n_mels =", n_mels)

    stft_and_display(y, n_fft=1024, hop_length=256, n_mels=n_mels, mel_scale=True, to_db_scale=True, show_shape=True)
for n_mels in range(50, 1050, 250):

    print("n_mels =", n_mels)

    stft_and_display(y, n_fft=8192, hop_length=256, n_mels=n_mels, mel_scale=True, to_db_scale=True, show_shape=True)
# Cleanup

! rm -fr fastai_audio