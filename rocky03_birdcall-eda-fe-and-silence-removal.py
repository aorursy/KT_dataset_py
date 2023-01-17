# import dependencies
import os
import numpy as np
import pandas as pd
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import folium
from folium import Marker,GeoJson,Choropleth, Circle
from folium.plugins import HeatMap, MarkerCluster
import librosa.display
from IPython.display import Audio

pd.set_option('display.max_columns', 50)
# import data
df_train = pd.read_csv("../input/birdsong-recognition/train.csv")
df_train.head
df_train.info()
df_train['ebird_code'].nunique()
df_train['species'].value_counts()
df_train['year'] = df_train['date'].apply(lambda x: x.split('-')[0])
df_train['month'] = df_train['date'].apply(lambda x: x.split('-')[1])
group_year = df_train.groupby(['year']).size().reset_index(name='counts')
group_year = group_year.iloc[3:]
group_month = df_train.groupby(['month']).size().reset_index(name='counts')



fig = make_subplots(rows=2, cols=1, subplot_titles = ('Number of recordings w.r.t year', 'Number of recordings w.r.t month'))

fig.append_trace(go.Bar(
    x=group_year['year'],
    y=group_year['counts'],
    #tickmode='linear'
), row=1, col=1)

fig.append_trace(go.Bar(
    x=group_month['month'],
    y=group_month['counts'],
), row=2, col=1)



fig.update_layout(height=1000, width=700, showlegend=False,  xaxis = dict(
        tickmode = 'linear',
    ), xaxis2 = dict(tickmode='linear'))
fig.show()
fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "pie"}]], subplot_titles = ('Distribution of Channels', 'Distribution of Sampling rate'))

group_ch = df_train.groupby(['channels']).size().reset_index(name='counts')
fig.append_trace(go.Pie(
    labels=group_ch['channels'],
    values=group_ch['counts'],
), row=1, col=1)

group_sr = df_train.groupby(['sampling_rate']).size().reset_index(name='counts')
fig.append_trace(go.Pie(
    labels=group_sr['sampling_rate'],
    values=group_sr['counts'],
), row=1, col=2)


fig.show()

map = folium.Map(location=[54, 15], tiles='cartodbpositron', zoom_start=5)
df_train = df_train[df_train["latitude"] != "Not specified"]

#drop nan values and convert latitude and longitude to float
df_no_nan = df_train.dropna(subset=['latitude','longitude'], how='any')
df_no_nan.latitude.astype(float)
df_no_nan.longitude.astype(float)

map_cluster = MarkerCluster()

# Add points to the map
for idx, row in df_no_nan.iterrows():
    map_cluster.add_child(Marker([row['latitude'], row['longitude']]))

map.add_child(map_cluster)

#Display map
map



audio_path = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'
x, sr = librosa.load(audio_path)
Audio(x, rate=sr)
audio_path = '../input/birdsong-recognition/train_audio/amepip/XC111040.mp3'
x, sr = librosa.load(audio_path)
Audio(x, rate=sr)
audio_path = '../input/birdsong-recognition/train_audio/banswa/XC138517.mp3'
x, sr = librosa.load(audio_path)
Audio(x, rate=sr)
audio_path = '../input/birdsong-recognition/train_audio/bkhgro/XC109305.mp3'
x, sr = librosa.load(audio_path)
Audio(x, rate=sr)
fig, ax = plt.subplots(4, figsize = (20, 9))
fig.suptitle('Waveplots', fontsize=16)
audio_path1 = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'
audio_path2 = '../input/birdsong-recognition/train_audio/amepip/XC111040.mp3'
audio_path3 = '../input/birdsong-recognition/train_audio/banswa/XC138517.mp3'
audio_path4 = '../input/birdsong-recognition/train_audio/bkhgro/XC109305.mp3'

y1, sr1 = librosa.load(audio_path1)
y2, sr2 = librosa.load(audio_path2)
y3, sr3 = librosa.load(audio_path3)
y4, sr4 = librosa.load(audio_path4)

librosa.display.waveplot(y=y1, sr=sr1, color = "#3371FF", ax=ax[0])
librosa.display.waveplot(y=y2 , sr=sr2, color = "#F7A81E", ax=ax[1])
librosa.display.waveplot(y=y3 , sr=sr3, color = "#2BF71E", ax=ax[2])
librosa.display.waveplot(y=y4 , sr=sr4, color = "#F71E6D", ax=ax[3])

# Visualize an STFT power spectrum

audio_path = '../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'
y, sr = librosa.load(audio_path)
plt.figure(figsize=(12, 8))
D = librosa.amplitude_to_db(librosa.stft(y))
plt.subplot(4, 2, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram')

# logarithmic scale

plt.subplot(4, 2, 2)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram')

#CQT scale

CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
plt.subplot(4, 2, 3)
librosa.display.specshow(CQT, y_axis='cqt_hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (Hz)')

CQT = librosa.amplitude_to_db(librosa.cqt(y, sr=sr), ref=np.max)
plt.subplot(4, 2, 4)
librosa.display.specshow(CQT, y_axis='cqt_note')
plt.colorbar(format='%+2.0f dB')
plt.title('Constant-Q power spectrogram (note)')

#Chromagram
C = librosa.feature.chroma_cqt(y=y, sr=sr)
plt.subplot(4, 2, 5)
librosa.display.specshow(C, y_axis='chroma')
plt.colorbar()
plt.title('Chromagram')

# Log power spectrogram
plt.subplot(4, 2, 6)
librosa.display.specshow(D, x_axis='time', y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log power spectrogram')

# let's zoom in 
n0 = 7000
n1 = 7100
plt.figure(figsize=(14, 5))
plt.plot(y[n0:n1])
zero_crossings = librosa.zero_crossings(y[n0:n1], pad=False)
zero_crossings.shape
print(sum(zero_crossings))
zcrs = librosa.feature.zero_crossing_rate(y)
print(zcrs.shape)
plt.figure(figsize=(14, 5))
plt.plot(zcrs[0])
spectral_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0]
spectral_centroid.shape
plt.figure(figsize=(14, 5))
plt.plot(spectral_centroid.T, label='Spectral centroid')
plt.ylabel('Hz')
plt.xticks([])
plt.xlim([0, spectral_centroid.shape[-1]])
plt.legend()
#  time variable for visualization
frames = range(len(spectral_centroid))
t = librosa.frames_to_time(frames)

# helper function to normalize the spectral centroid for visualization

def normalize(y, axis=0):
    return sklearn.preprocessing.minmax_scale(y, axis=axis)

spectral_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]
librosa.display.waveplot(y, sr=sr, alpha=0.4)
plt.plot(t, normalize(spectral_rolloff), color='r')
audio_path = '../input/birdsong-recognition/train_audio/amecro/XC114552.mp3'
y, sr = librosa.load(audio_path)
Audio(y, rate=sr)
db = librosa.core.amplitude_to_db(y)
mean_db = np.abs(db).mean()
std_db = db.std()
x_split = librosa.effects.split(y=y, top_db = mean_db - std_db)
silence_removed = []
for i in x_split:
    silence_removed.extend(y[i[0]:i[1]])
silence_removed = np.array(silence_removed)
Audio(silence_removed, rate=sr)

