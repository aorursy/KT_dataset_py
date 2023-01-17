import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import math

from math import *



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objects as go

from ipywidgets import widgets

from ipywidgets import *



init_notebook_mode(connected=True)

%matplotlib inline
import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)
train = pd.read_csv("../input/birdsong-recognition/train.csv",delimiter=",",encoding="latin", engine='python')

test = pd.read_csv("../input/birdsong-recognition/test.csv",delimiter=",",encoding="latin", engine='python')

audio_summary = pd.read_csv("../input/birdsong-recognition/example_test_audio_summary.csv",delimiter=",",encoding="latin", engine='python')

audio_metadata = pd.read_csv("../input/birdsong-recognition/example_test_audio_metadata.csv",delimiter=",",encoding="latin", engine='python')
train.head(5)
plt.figure(figsize=(20,10))

sns.heatmap(train.isna(), cbar=False)
missing_rate_train = (train.isna().sum()/train.shape[0]).sort_values()

nb_missing = train.isna().sum().sort_values()

print(f'{"Variable" :-<25} {"missing_rate_train":-<25} {"Number of missing values":-<25}')

for n in range(len(missing_rate_train)):

    print(f'{missing_rate_train.index[n] :-<25} {missing_rate_train[n]:-<25} {nb_missing[n]:-<25}')
train.columns
len(train["species"].unique())
rate = train["species"].value_counts().sort_values()/264

print(f'{"Target" :-<40} {"rate":-<20}')

for n in range(len(rate)):

    print(f'{rate.index[n] :-<40} {rate[n]}')
train["species"].value_counts().sort_values().iplot(kind="bar",)
longitude = pd.to_numeric(train['longitude'], errors='coerce')

latitude = pd.to_numeric(train['latitude'], errors='coerce')

df = pd.concat([longitude,latitude],axis=1)
import folium

from folium import Choropleth, Circle, Marker

from folium.plugins import HeatMap, MarkerCluster

f = folium.Figure(width=1000, height=500)



longitude = pd.to_numeric(train['longitude'], errors='coerce')

latitude = pd.to_numeric(train['latitude'], errors='coerce')

df = pd.concat([longitude,latitude],axis=1).dropna()

m = folium.Map(location=[40, 0], zoom_start=2).add_to(f)

# Add a heatmap to the base map

HeatMap(data=df[['latitude', 'longitude']], radius=10).add_to(m)

m
train['playback_used'].fillna('Missing',inplace=True)

labels=train['playback_used'].value_counts().index

values=train['playback_used'].value_counts().values

fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial'

                            )])

fig.show()
train['date'] = train['date'].apply(pd.to_datetime,format='%Y-%m-%d', errors='coerce')

train['date'].value_counts().plot(figsize=(25, 6))
plt.figure(figsize=(25, 6))

ax = sns.countplot(train['date'].dt.year.dropna().apply(lambda x : int(x)), palette="hls")

train["length"].value_counts().sort_values().iplot(kind="bar",)
train["author"].value_counts().sort_values().iplot(kind="bar",)
train["ebird_code"].value_counts().sort_values().iplot(kind="bar")
from IPython.display import YouTubeVideo



YouTubeVideo('MhOdbtPhbLU', width=800, height=300)
import warnings

warnings.filterwarnings('ignore')
import librosa

import librosa.display

audio_data = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'

x , sr = librosa.load(audio_data)

print(x.shape, sr)
librosa.load(audio_data,sr)
import IPython.display as ipd

ipd.Audio(audio_data)
from random import sample 

import matplotlib.pyplot as plt

from matplotlib import colors as mcolors

colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)

                for name, color in colors.items())

colorsName = [name for hsv, name in by_hsv]
class AudioProcessing:

    

    def ReadAudio(self,ebird_code,filename):

        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename

        x , sr = librosa.load(audio_file)

        return x,sr

    

    def LoadAudio(self,audio_file,sr):

        return librosa.load(audio_file,sr)

        

    def PlayingAudio(self,ebird_code,filename):

        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename

        x , sr = librosa.load(audio_file)

        librosa.load(audio_file,sr)

        return ipd.Audio(audio_file)

                         

    def DisplayWave(self,ebird_code,filename):

        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename

        y, sr = librosa.load(audio_file)

        whale_song, _ = librosa.effects.trim(y)

        plt.figure(figsize=(12, 4))

        librosa.display.waveplot(whale_song, sr=sr)

        plt.show()

                         

    def DisplaySpectogram(self,ebird_code,filename):

        audio_file = "../input/birdsong-recognition/train_audio" + "/" + ebird_code + "/" + filename

        x , sr = librosa.load(audio_file)

        Xdb = librosa.amplitude_to_db(abs(librosa.stft(x)))

        plt.figure(figsize=(12, 4))

        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

        plt.colorbar()

        plt.show()               

    def PlotSampleWave(self,nrows,captions,df):

        ncols=1

        f, ax = plt.subplots(nrows,ncols=ncols,figsize=(ncols*12,nrows*3))

        i = 0

        colors = sample(colorsName,nrows)

        for c in captions:

            samples = df[df['ebird_code']==c]['filename'].sample(ncols).values

            audio_file = "../input/birdsong-recognition/train_audio" + "/" + c + "/" + samples[0]

            y, sr = librosa.load(audio_file)

            whale_song, _ = librosa.effects.trim(y)

            librosa.display.waveplot(whale_song, sr=sr, color = colors[i],ax=ax[i])

            i = i + 1

        for i, name in zip(range(nrows), captions):

            ax[i].set_ylabel(name, fontsize=15)

        plt.tight_layout()

        plt.show()

    

    def PlotSampleSpectrogram(self,nrows,captions,df):

        ncols=1

        f, ax = plt.subplots(nrows,ncols=ncols,figsize=(ncols*12,nrows*3))

        i = 0

        colors = sample(colorsName,nrows)

        for c in captions:

            samples = df[df['ebird_code']==c]['filename'].sample(ncols).values

            audio_file = "../input/birdsong-recognition/train_audio" + "/" + c + "/" + samples[0]

            x, sr = librosa.load(audio_file)

            X = librosa.stft(x)

            Xdb = librosa.amplitude_to_db(abs(X))

            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz',ax=ax[i])

            i = i + 1

        for i, name in zip(range(nrows), captions):

            ax[i].set_ylabel(name, fontsize=15)

        plt.tight_layout()

        plt.show()

        

    def PlotSampleMelSpectrogram(self,nrows,captions,df):

        ncols=1

        f, ax = plt.subplots(nrows,ncols=ncols,figsize=(ncols*12,nrows*3))

        i = 0

        colors = sample(colorsName,nrows)

        for c in captions:

            samples = df[df['ebird_code']==c]['filename'].sample(ncols).values

            audio_file = "../input/birdsong-recognition/train_audio" + "/" + c + "/" + samples[0]

            x, sr = librosa.load(audio_file)

            S = librosa.feature.melspectrogram(x, sr)

            log_S = librosa.power_to_db(S, ref=np.max)



            librosa.display.specshow(log_S, sr = sr, hop_length = 500, x_axis = 'time', y_axis = 'log', cmap = 'rainbow', ax=ax[i])

            i = i + 1

        for i, name in zip(range(nrows), captions):

            ax[i].set_ylabel(name, fontsize=15)

        plt.tight_layout()

        plt.show()

            
N = 5

ebird_code_simple = sample(list(train["ebird_code"].unique()),N)

AudioProcessing().PlotSampleWave(nrows=N,captions=ebird_code_simple,df=train)
AudioProcessing().PlotSampleSpectrogram(nrows=N,captions=ebird_code_simple,df=train)
AudioProcessing().PlotSampleMelSpectrogram(nrows=N,captions=ebird_code_simple,df=train)