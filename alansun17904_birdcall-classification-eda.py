import os

import numpy as np

import pandas as pd

import librosa

import librosa.display

import plotly.express as px

import plotly.graph_objects as go

import folium

import IPython

import matplotlib.pyplot as plt
print('\n'.join(os.listdir('/kaggle/input/birdsong-recognition')))

path = '/kaggle/input/birdsong-recognition'

TRAIN_PATH = os.path.join(path, 'train_audio')

TEST_PATH = os.path.join(path, 'example_test_audio')
train_meta = pd.read_csv(os.path.join(path, 'train.csv'))

test_meta = pd.read_csv(os.path.join(path, 'test.csv'))
rating_count = train_meta.groupby('rating')['xc_id'].count().reset_index()

rating_count.rename(columns={'xc_id': 'Count', 'rating': 'Rating'}, inplace=True)

fig = px.bar(rating_count, x='Rating', y='Count', title='Bar Plot of Number of Audio Recordings in Each Rating Category')

fig.show()
labels = train_meta.groupby(['species'])['xc_id'].count().reset_index()

labels.rename(columns={'xc_id': 'Count', 'species': 'Species'}, inplace=True)

labels.sort_values(by=['Count'], inplace=True, ascending=True)

fig = px.bar(labels, x='Species', y='Count', hover_data=['Species'], color='Species')

fig.show()
fig = px.histogram(train_meta['duration'].reset_index(), x='duration', labels={'count': 'Count', 'duration': 'Duration'},

                   title='Distribution of Duration of Audio Files')

fig.show()
train_meta.sampling_rate = train_meta.sampling_rate.apply(lambda sr: int(sr.split(' ')[0])).astype(np.uint16)

fig = px.histogram(train_meta['sampling_rate'].reset_index(), x='sampling_rate', labels={'count': 'Count', 'sampling_rate': 'Sampling Rate'},

                   title='Distribution of Sampling Rate of Audio Files')

fig.show()
sample1 = train_meta.iloc[528]
IPython.display.Audio(filename=os.path.join(TRAIN_PATH, 

                                            os.path.join(sample1['ebird_code'],sample1['filename'])))
sample1
y, sr = librosa.load(os.path.join(os.path.join(TRAIN_PATH, sample1['ebird_code']), sample1['filename']), 

                     sr=sample1['sampling_rate'])

sample1_audio, _ = librosa.effects.trim(y)

time = [v/sample1['sampling_rate'] for v in range(len(sample1_audio))]

fig = px.line({'Time': time, 'Intensity': sample1_audio}, x='Time', y='Intensity', title='Wave Plot of Sample1 Audio')

fig.show()
n_fft = 2048

D = np.abs(librosa.stft(sample1_audio[:n_fft], n_fft=n_fft, hop_length=n_fft+1))

fig = px.line({'Frequency': np.array(range(len(D))), 'Magnitude': D.flatten()}, x='Frequency', y='Magnitude', 

              title='Fourier Transformation of Sample1 Audio')

fig.show()
hop_length = 512

n_fft = 2048

D = np.abs(librosa.stft(sample1_audio, n_fft=n_fft,  hop_length=hop_length))

librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear');

plt.colorbar();
DB = librosa.amplitude_to_db(D, ref=np.max)

librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');

plt.colorbar(format='%+2.0f dB');