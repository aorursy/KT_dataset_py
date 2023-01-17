import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
from colorama import Fore, Back, Style 
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import xgboost
from plotly.offline import plot, iplot, init_notebook_mode
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.express as px
from statsmodels.formula.api import ols
import plotly.graph_objs as gobj
init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
import plotly.figure_factory as ff

%matplotlib inline
ds=pd.read_csv("../input/birdsong-recognition/train.csv")
fig = go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = ds['longitude'],
        lat = ds['latitude'],
        text = ds['location'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'square',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Blues',
            cmin = 0,
            color = ds['duration'],
            cmax = ds['duration'].max(),
            colorbar_title="Duration"
        )))

fig.update_layout(
        title = 'Recorded Place all over the world',
        geo = dict(
            scope='world',
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5
        ),
    )
fig.show()
val=ds["pitch"].value_counts()
nt=["Not specified"]
lvl=["level"] 
bt=["both"]
dc=["decreasing"]
inc=["increasing"]

fig = go.Figure(data=[
    go.Bar(name='Not specified', x=nt, y=[val["Not specified"]]),
    go.Bar(name='Level', x=lvl, y=[val["level"]]),
    go.Bar(name='Both', x=bt, y=[val["both"]]),
    go.Bar(name='decreasing', x=dc, y=[val["decreasing"]]),
    go.Bar(name='increasing', x=inc, y=[val["increasing"]])
    
    
])
# Change the bar mode
fig.update_layout(barmode='group',title="Pitch Labels")
fig.show()
import librosa
audio_path = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'
x , sr = librosa.load(audio_path)
print(type(x), type(sr))
print(x.shape, sr)

librosa.load(audio_path, sr=None)

import IPython.display as ipd
ipd.Audio(audio_path)
%matplotlib inline
import matplotlib.pyplot as plt
import librosa.display
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
X = librosa.stft(x)
Xdb = librosa.amplitude_to_db(abs(X))
plt.figure(figsize=(14, 5))
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
plt.colorbar()
x, sr = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3')
#Plot the signal:
plt.figure(figsize=(14, 5))
librosa.display.waveplot(x, sr=sr)
n0 = 9000
n1 = 9100
plt.figure(figsize=(14, 5))
plt.plot(x[n0:n1])
plt.grid()
import sklearn
spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]
spectral_centroids.shape
(775,)
# Computing the time variable for visualization
frames = range(len(spectral_centroids))
t = librosa.frames_to_time(frames)
# Normalising the spectral centroid for visualisation
def normalize(x, axis=0):
    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_centroids), color='r')
spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]
librosa.display.waveplot(x, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
x, fs = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3')
librosa.display.waveplot(x, sr=sr)
mfccs = librosa.feature.mfcc(x, sr=fs)
print(mfccs.shape)
(20, 97)
#Displaying  the MFCCs:
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
import sklearn
mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
print(mfccs.mean(axis=1))
print(mfccs.var(axis=1))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
# Loadign the file
x, sr = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3')
hop_length = 512
chromagram = librosa.feature.chroma_stft(x, sr=sr, hop_length=hop_length)
plt.figure(figsize=(15, 5))
librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm')
