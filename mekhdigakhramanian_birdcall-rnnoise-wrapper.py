!pip install git+https://github.com/Desklop/RNNoise_Wrapper
import librosa
import librosa.display as display
import pandas as pd
import soundfile as sf

from IPython.display import Audio
from pathlib import Path

import IPython
import matplotlib.pyplot as plt
import scipy.signal
noise_path='../input/birdsong-resampled-train-audio-00/aldfly/XC179417.wav'
noise_clip , sr = librosa.load(noise_path, sr=32000)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(noise_clip)
ax.title.set_text('Orginal')
IPython.display.Audio(data=noise_clip, rate=sr)
denoiser = RNNoise()

audio = denoiser.read_wav('../input/birdsong-resampled-train-audio-00/aldfly/XC179417.wav')
filtered_audio = denoiser.filter(audio)
denoiser.write_wav('test_denoised.wav', filtered_audio)
noise_path='./test_denoised.wav'
noise_clip , sr = librosa.load(noise_path, sr=32000)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(noise_clip)
ax.title.set_text('Denoised')
IPython.display.Audio(data=noise_clip, rate=sr)
audio = denoiser.read_wav('../input/birdsong-resampled-train-audio-00/aldfly/XC179417.wav')

filtered_audio = b''
buffer_size_ms = 10

for i in range(buffer_size_ms, len(audio), buffer_size_ms):
    filtered_audio += denoiser.filter(audio[i-buffer_size_ms:i].raw_data, sample_rate=audio.frame_rate)
if len(audio) % buffer_size_ms != 0:
    filtered_audio += denoiser.filter(audio[len(audio)-(len(audio)%buffer_size_ms):].raw_data, sample_rate=audio.frame_rate)

denoiser.write_wav('test_denoised_f.wav', filtered_audio, sample_rate=audio.frame_rate)
noise_path='./test_denoised_f.wav'
noise_clip , sr = librosa.load(noise_path, sr=32000)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(noise_clip)
ax.title.set_text('Denoised - Filtered')
IPython.display.Audio(data=noise_clip, rate=sr)