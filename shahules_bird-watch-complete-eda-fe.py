!pip install librosa
from plotly.subplots import make_subplots

import plotly.graph_objects as go

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

import IPython.display as ipd

import plotly.express as px

import librosa.display

import pandas as pd

import numpy as  np

import librosa

import warnings

import IPython

import os



plt.style.use("ggplot")
warnings.filterwarnings(action='ignore')


train = pd.read_csv("../input/birdsong-recognition/train.csv")
train.info()
train.head(3)
print("Train dataset has {} rows and {} columns".format(*train.shape))
print("There are {} unique species of birds in train dataset".format(train.species.nunique()))
species=train.species.value_counts()


fig = go.Figure(data=[

    go.Bar(y=species.values, x=species.index,marker_color='deeppink')

])



fig.update_layout(title='Distribution of Bird Species')

fig.show()

country = train.country.value_counts()[:20]

fig = go.Figure(data=[

    go.Bar(x=country.index, y=country.values,marker_color='deeppink')

])



fig.update_layout(title='Countries from which data is obtained')

fig.show()
plt.figure(figsize=(12, 8))

train['date'].value_counts().sort_index().plot(color='pink',alpha=1)




hist_data = pd.to_datetime(train.time,errors='coerce').dropna().dt.hour.values.tolist()

fig = go.Figure(data=[go.Histogram(x=hist_data, histnorm='probability',marker_color='deeppink')])

fig.update_layout(title='Time of the day at which data is obtained')



fig.show()







hist_data = train.duration.values.tolist()

fig = go.Figure(data=[go.Histogram(x=hist_data,marker_color='deeppink')])

fig.update_layout(title='Duration of the observation')



fig.show()



hist_data = train.rating.values.tolist()

fig = go.Figure(data=[go.Histogram(x=hist_data,marker_color='deeppink')])

fig.update_layout(title='Rating of the observation')



fig.show()

colors = ['gold', 'mediumturquoise', 'darkorange', 'lightgreen']

df = train.bird_seen.value_counts()

fig = px.pie(df,df.index,df.values,labels={'index':'Bird Seen'})

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title='Bird Seen')



fig.show()


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

df = train.volume.value_counts()

fig.add_trace(go.Pie(labels=df.index, values=df.values, name="Volume"),

              1, 1)



df = train.pitch.value_counts()

fig.add_trace(go.Pie(labels=df.index ,values=df.values, name="Pitch"),

              1, 2)



# Use `hole` to create a donut-like pie chart

fig.update_traces(hole=.4, hoverinfo="label+percent+name")

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.update_layout(

    title_text="Volume and Pitch of Observation",

    # Add annotations in the center of the donut pies.

    annotations=[dict(text='Volume', x=0.18, y=0.5, font_size=20, showarrow=False),

                 dict(text='Pitch', x=0.82, y=0.5, font_size=20, showarrow=False)])

fig.show()
rec = train.sampling_rate.value_counts()

fig = go.Figure(data=[

    go.Bar(x=rec.index, y=rec.values,marker_color='deeppink')

])



fig.update_layout(title='Top Recordists')

fig.show()
rec = train.channels.value_counts()

fig = go.Figure(data=[

    go.Bar(x=rec.index, y=rec.values,marker_color='deeppink')

])



fig.update_layout(title='Top Recordists')

fig.show()
df=train.length.value_counts()

fig = px.pie(df,df.index,df.values,labels={'index':'length of audio'})

fig.update_layout(title='Length of audio signal')

fig.update_traces(hoverinfo='label+percent', textinfo='value', textfont_size=20,

                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))



fig.show()
df=train.groupby(['latitude','longitude'],as_index=False)['ebird_code'].agg('count')
df=df[df.latitude!='Not specified']

fig = go.Figure()

fig.add_trace(go.Scattergeo(

        lon = df['longitude'],

        lat = df['latitude'],

        text = df['ebird_code'],

        marker = dict(

            size = df['ebird_code'],

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        )))





fig.update_layout(

        title_text = 'Bird Samples collected From Parts of World',

        showlegend = True,

        geo = dict(

            landcolor = 'rgb(217, 217, 217)',

        )

    )



fig.show()

fig = go.Figure()

fig.add_trace(go.Scattergeo(

        locationmode = 'USA-states',

        lon = df['longitude'],

        lat = df['latitude'],

        text = df['ebird_code'],

        marker = dict(

            size = df['ebird_code'],

            line_color='rgb(40,40,40)',

            line_width=0.5,

            sizemode = 'area'

        )))





fig.update_layout(

        title_text = 'Bird Samples collected From USA',

        showlegend = True,

        geo = dict(

            scope = 'usa',

            landcolor = 'rgb(217, 217, 217)',

        )

    )



fig.show()

path="../input/birdsong-recognition/train_audio/"

birds=train.ebird_code.unique()[:6]

file=train[train.ebird_code==birds[0]]['filename'][0]


for i in range(0,2):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    print(birds[i])

    IPython.display.display(ipd.Audio(audio_path))



plt.figure(figsize=(17,20 ))





for i in range(0,6):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(6,2,i+1)

    x , sr = librosa.load(audio_path)

    librosa.display.waveplot(x, sr=sr,color='r')

    plt.gca().set_title(birds[i])

    plt.gca().get_xaxis().set_visible(False)



plt.figure(figsize=(17,20 ))





for i in range(0,6):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(6,2,i+1)

    x , sr = librosa.load(audio_path)

    x = librosa.stft(x)

    Xdb = librosa.amplitude_to_db(abs(x))

    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

    plt.gca().set_title(birds[i])

    plt.gca().get_xaxis().set_visible(False)

    plt.colorbar()
import sklearn

# Normalising the spectral centroid for visualisation

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)


plt.figure(figsize=(17,20 ))





for i in range(0,6):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(6,2,i+1)

    x , sr = librosa.load(audio_path)

    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

    frames = range(len(spectral_centroids))

    t = librosa.frames_to_time(frames)

    librosa.display.waveplot(x, sr=sr, alpha=0.4)

    plt.plot(t, normalize(spectral_centroids), color='b')

    plt.gca().set_title(birds[i])

    plt.gca().get_xaxis().set_visible(False)

    



plt.figure(figsize=(17,20 ))





for i in range(0,6):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(6,3,i+1)

    x , sr = librosa.load(audio_path)

    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

    frames = range(len(spectral_centroids))

    t = librosa.frames_to_time(frames)

    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

    librosa.display.waveplot(x, sr=sr, alpha=0.4)

    plt.plot(t, normalize(spectral_rolloff), color='r')

    plt.gca().set_title(birds[i])

    plt.gca().get_xaxis().set_visible(False)

    


plt.figure(figsize=(17,20 ))





for i in range(0,6):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(6,3,i+1)

    x , sr = librosa.load(audio_path)

    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

    frames = range(len(spectral_centroids))

    t = librosa.frames_to_time(frames)

    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]

    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]

    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]

    librosa.display.waveplot(x, sr=sr, alpha=0.4)

    plt.plot(t, normalize(spectral_bandwidth_2), color='r')

    plt.plot(t, normalize(spectral_bandwidth_3), color='g')

    plt.plot(t, normalize(spectral_bandwidth_4), color='y')

    plt.gca().set_title(birds[i])

    plt.gca().get_xaxis().set_visible(False)

    plt.legend(('p = 2', 'p = 3', 'p = 4'))
x , sr = librosa.load(audio_path)

plt.figure(figsize=(14, 5))

librosa.display.waveplot(x, sr=sr)

# Zooming in

n0 = 9000

n1 = 9100

plt.figure(figsize=(14, 5))

plt.plot(x[n0:n1])

plt.grid()
zero_crossings = librosa.zero_crossings(x[n0:n1], pad=False)

print(sum(zero_crossings))
plt.figure(figsize=(17, 20))



for i in range(0,6):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(6,2,i+1)

    x , sr = librosa.load(audio_path)

    mfccs = librosa.feature.mfcc(x, sr=sr)

    librosa.display.specshow(mfccs, sr=sr, x_axis='time')

    plt.gca().set_title(birds[i])

    plt.gca().get_xaxis().set_visible(False)

    
plt.figure(figsize=(17, 20))



for i in range(0,6):

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(6,3,i+1)

    x , sr = librosa.load(audio_path)

    chromagram = librosa.feature.chroma_stft(x, sr=sr)

    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')

    plt.gca().set_title(birds[i])

    plt.gca().get_xaxis().set_visible(False)

    
fig=plt.figure(figsize=(15,15))

k=1

for i in range(5):

    

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(5,3,k)

    k+=1

    x , sr = librosa.load(audio_path)

    librosa.display.waveplot(x, sr=sr)

    plt.gca().set_title('Spectral Centroid')

    plt.gca().set_ylabel(birds[i])

    plt.gca().get_xaxis().set_visible(False)



    plt.subplot(5,3,k)

    k+=1

    spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

    frames = range(len(spectral_centroids))

    t = librosa.frames_to_time(frames)

    spectral_rolloff = librosa.feature.spectral_rolloff(x+0.01, sr=sr)[0]

    librosa.display.waveplot(x, sr=sr, alpha=0.4)

    plt.plot(t, normalize(spectral_rolloff), color='r')

    plt.gca().set_title('Spectral Rolloff ')

    plt.gca().get_xaxis().set_visible(False)



    plt.subplot(5,3,k)

    k+=1

    #spectral_centroids = librosa.feature.spectral_centroid(x, sr=sr)[0]

    #frames = range(len(spectral_centroids))

    #t = librosa.frames_to_time(frames)

    spectral_bandwidth_2 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr)[0]

    spectral_bandwidth_3 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=3)[0]

    spectral_bandwidth_4 = librosa.feature.spectral_bandwidth(x+0.01, sr=sr, p=4)[0]

    librosa.display.waveplot(x, sr=sr, alpha=0.4)

    plt.plot(t, normalize(spectral_bandwidth_2), color='r')

    plt.plot(t, normalize(spectral_bandwidth_3), color='g')

    plt.plot(t, normalize(spectral_bandwidth_4), color='y')

    plt.gca().set_title('Spectral Bandwidth')

    plt.gca().get_xaxis().set_visible(False)

    plt.legend(('p = 2', 'p = 3', 'p = 4'))



    

#plt.gca().set_title('Comparing audio features for bird species')

plt.tight_layout()

plt.show()
fig=plt.figure(figsize=(15,15))

k=1

for i in range(5):

    

    file=train[train.ebird_code==birds[i]]['filename'].values[0]

    audio_path=os.path.join(path,birds[i],file)

    plt.subplot(5,3,k)

    k+=1

    x , sr = librosa.load(audio_path)

    s = librosa.stft(x)

    Xdb = librosa.amplitude_to_db(abs(s))

    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

    plt.gca().set_title('Spectrogram')

    plt.gca().set_ylabel(birds[i])

    plt.gca().get_xaxis().set_visible(False)



    plt.subplot(5,3,k)

    k+=1

    mfccs = librosa.feature.mfcc(x, sr=sr)

    librosa.display.specshow(mfccs, sr=sr, x_axis='time')

    plt.gca().set_title('MFFC features ')

    plt.gca().get_xaxis().set_visible(False)



    plt.subplot(5,3,k)

    k+=1

    chromagram = librosa.feature.chroma_stft(x, sr=sr)

    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', cmap='coolwarm')

    plt.gca().set_title('Chroma feature')

    plt.gca().get_xaxis().set_visible(False)

  



    

#fig.suptitle('Comparing audio features for bird species')

plt.tight_layout()

plt.show()