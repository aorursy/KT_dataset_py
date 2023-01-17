import io

import os

from zipfile import ZipFile



import numpy as np

import pandas as pd

import librosa as lr

from tqdm import tqdm



os.listdir("../input")
df = pd.read_csv('../input/train_curated.csv', index_col='fname')

df.index = df.index.str.replace('.wav', '.npy')

binary_indicators = df.labels.str.get_dummies(',')

binary_indicators.head()
from librosa.display import specshow





def preprocess(wavfile):



    # Load roughly 8 seconds of audio.

    samples = 512*256 - 1

    samplerate = 16000

    waveform = lr.load(wavfile, samplerate, duration=samples/samplerate)[0]



    # Loop too short audio clips.

    if len(waveform) < samples:

        waveform = np.pad(waveform, (0, samples - len(waveform)), mode='wrap')



    # Convert audio to log-mel spectrogram.

    spectrogram = lr.feature.melspectrogram(waveform, samplerate, n_mels=256)

    spectrogram = lr.power_to_db(spectrogram)

    spectrogram = spectrogram.astype(np.float32)



    return spectrogram





sample = df.sample()

spectrogram = preprocess('../input/train_curated/0006ae4e.wav')

ax = specshow(spectrogram, y_axis='mel', x_axis='time')

ax.set_title('Example spectrogram')

spectrogram.shape
with ZipFile('dataset.zip', 'w') as z:



    with io.StringIO() as f:

        binary_indicators.to_csv(f)

        z.writestr('index.csv', f.getvalue())



    d = '../input/train_curated'

    for n in tqdm(os.listdir(d), desc='Zipping spectrograms'):

        wavfile = os.path.join(d, n)

        spectrogram = preprocess(wavfile)

        with io.BytesIO() as f:

            np.save(f, spectrogram)

            z.writestr(n.replace('.wav', '.npy'), f.getvalue())