from __future__ import unicode_literals



!pip install youtube_dl

!apt install -y ffmpeg



import time

import youtube_dl
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# feature extractoring and preprocessing data

import librosa

import librosa.display

import matplotlib.pyplot as plt

%matplotlib inline

from PIL import Image

import pathlib

import csv



# Preprocessing

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, StandardScaler



#Keras

import keras



import warnings

warnings.filterwarnings('ignore')
header = 'filename tempo chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'

for i in range(1, 21):

    header += f' mfcc{i}'

header += ' label'

header = header.split()
file = open('data.csv', 'w', newline='')

with file:

    writer = csv.writer(file)

    writer.writerow(header)

genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()

for g in genres:

    print(f'Genre:{g} started')

    start = time.clock()

    for filename in os.listdir(f'/kaggle/input/gtzan-genre-collection/genres/{g}'):

        songname = f'/kaggle/input/gtzan-genre-collection/genres/{g}/{filename}'

        y, sr = librosa.load(songname, mono=True, duration=30,sr=None)

        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

        rms = librosa.feature.rms(y=y)

        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

        zcr = librosa.feature.zero_crossing_rate(y)

        mfcc = librosa.feature.mfcc(y=y, sr=sr)

        tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])

        to_append = f'{filename} {tempo} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    

        for e in mfcc:

            to_append += f' {np.mean(e)}'

        to_append += f' {g}'

        file = open('data.csv', 'a', newline='')

        with file:

            writer = csv.writer(file)

            writer.writerow(to_append.split())

    print(f'Genre:{g} completed, took {time.clock()-start} seconds')

data = pd.read_csv('data.csv')

data.head()
data.shape
# Dropping unneccesary columns

data = data.drop(['filename'],axis=1)
genre_list = data.iloc[:, -1]

encoder = LabelEncoder()

y = encoder.fit_transform(genre_list)

print(np.unique(y))
scaler = StandardScaler()

X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print('Train/test split:', len(y_train), len(y_test))
from catboost import Pool, CatBoostClassifier, cv, CatBoostRegressor

#let us make the catboost model, use_best_model params will make the model prevent overfitting

model = CatBoostClassifier(iterations=400,

                           learning_rate=0.05,

                           l2_leaf_reg=3.5,

                           depth=8,

                           rsm=0.98,

                           loss_function='MultiClass',

                           eval_metric='AUC',

                           use_best_model=True,

                           random_seed=42)



#now just to make the model to fit the data

model.fit(X_train,y_train,cat_features=[],eval_set=(X_test,y_test))





#last let us make the submission,note that you have to make the pred to be int!

pred = model.predict_proba(X_test)

preds= pred[:,1]
songname = f'./ytdl/sample.mp3'

try:

    os.remove(songname)

except Exception as e:

    print('Unable to remove:', e)

    

ydl_opts = {

    'format': 'bestaudio/best',

    'postprocessors': [{

        'key': 'FFmpegExtractAudio',

        'preferredcodec': 'mp3',

        'preferredquality': '192',

    }],

    'outtmpl': songname

}



with youtube_dl.YoutubeDL(ydl_opts) as ydl:

    ydl.download(['https://www.youtube.com/watch?v=NlprozGcs80'])

    

song_pd = pd.DataFrame(data.columns)

y, sr = librosa.load(songname, mono=True, duration=30)

chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

rms = librosa.feature.rms(y=y)

spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)

spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)

zcr = librosa.feature.zero_crossing_rate(y)

mfcc = librosa.feature.mfcc(y=y, sr=sr)

tempo = float(librosa.beat.tempo(y=y, sr=sr)[0])

row = [tempo, np.mean(chroma_stft), np.mean(rms), np.mean(spec_cent), np.mean(spec_bw), np.mean(rolloff), np.mean(zcr)]

for e in mfcc:

    row.append(np.mean(e))



X_test = np.asarray(row)

print('TYPE:', genres[model.predict(X_test)[0]])



plt.figure(figsize=(12, 8))

D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

plt.subplot(4, 2, 1)

librosa.display.specshow(D, y_axis='linear')

plt.colorbar(format='%+2.0f dB')

plt.title('Linear-frequency power spectrogram')