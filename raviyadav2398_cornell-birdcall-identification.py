import seaborn as sns

import matplotlib.pyplot as plt

from plotly.offline import init_notebook_mode, iplot

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

# Map 1 library

import plotly.express as px



# Map 2 libraries

import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon

# Librosa Libraries

import librosa

import librosa.display

import IPython.display as ipd

import sklearn

import warnings

warnings.filterwarnings('ignore')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path="/kaggle/input/birdsong-recognition/"

audio="/kaggle/input/birdsong-recognition/train_audio/"

train=pd.read_csv(path+'train.csv')

display('shape of train',train.shape)

display(train.head())
train.keys()
test=pd.read_csv(path+'test.csv')

display('shape of test Dataset',test.shape)

display(test.head())
train.info()
# Create some time features

train['year']=train['date'].apply(lambda x:x.split('-')[0])

train['month']=train['date'].apply(lambda x:x.split('-')[1])

train['day_of_month']=train['date'].apply(lambda x:x.split('-')[2])

# number of unique species in dataset

len(train['species'].unique())
#ebird code of bird species is unique or not

display(train['ebird_code'].is_unique)

#ebird code of bird species

display(list(train['ebird_code'].value_counts().head(15).reset_index()['index'])

)

display(len(train['ebird_code'].value_counts()))
plt.figure(figsize=(16,6))

ax=sns.countplot(train['year'],palette="hls")

plt.title("Year of the Audio Files Registration",fontsize=20)

plt.xticks(rotation=90,fontsize=13)

plt.yticks(fontsize=15)

plt.ylabel("Frequency",fontsize=14)

plt.xlabel("")
plt.figure(figsize=(16,6))

ax=sns.countplot(train['month'],palette="hls")

plt.title("Month of the Audio Files Registration",fontsize=20)

plt.xticks(rotation=90,fontsize=13)

plt.yticks(fontsize=15)

plt.ylabel("Frequency",fontsize=14)

plt.xlabel("")
#total number of people who provided the reordings 

train['recordist'].nunique()
# Top 10 recordist in terms of the number of recording done

train['recordist'].value_counts()[:10].sort_values().iplot(kind='bar',color="#2750BF")
#Check whether playback used or not

train['playback_used'].fillna('Not Defined',inplace=True)

train['playback_used'].value_counts().sort_values().iplot(kind='bar',color="#2750BF")
#check whether pitch is unique or not

display(train['pitch'].is_unique)

#length of pitch in the dataset

display(len(train['pitch'].value_counts()))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train['pitch'], palette="hls", order = train['pitch'].value_counts().index)





plt.title("Pitch (quality of sound - how high/low was the tone)", fontsize=16)

plt.xticks(fontsize=16)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("");

# check whether type of call is unique or not

display(len(train['type'].value_counts()))

train['type'].value_counts().iplot(kind='bar',color="#2750BF")
## Create a new variable type by exploding all the values

adjusted_type = train['type'].apply(lambda x: x.split(',')).reset_index().explode("type")



# Strip of white spaces and convert to lower chars

adjusted_type = adjusted_type['type'].apply(lambda x: x.strip().lower()).reset_index()

adjusted_type['type'] = adjusted_type['type'].replace('calls', 'call')



# Create Top 15 list with song types

top_15 = list(adjusted_type['type'].value_counts().head(15).reset_index()['index'])

data = adjusted_type[adjusted_type['type'].isin(top_15)]

plt.figure(figsize=(16, 6))

ax = sns.countplot(data['type'], palette="hls", order = data['type'].value_counts().index)





plt.title("Top 15 Song Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
plt.figure(figsize=(16, 6))

ax = sns.countplot(train['rating'], palette="hls", order = train['rating'].value_counts().index)





plt.title("Rating", fontsize=16)

plt.xticks(fontsize=16)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("");
# Create data

data = train['bird_seen'].value_counts().reset_index()

plt.figure(figsize=(16, 6))

ax = sns.barplot(x = 'bird_seen', y = 'index', data = data, palette="hls")





plt.title("Song was heard, but was Bird Seen?", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
train['country'].nunique()
train['country'].value_counts()
top10=list(train['country'].value_counts().head(10).reset_index()['index'])

top10

data=train[train['country'].isin(top10)]

plt.figure(figsize=(16, 6))

ax = sns.countplot(data['country'], palette='hls', order = data['country'].value_counts().index)





plt.title("Top 10 Countries with most Recordings", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Import gapminder data, where we have country and iso ALPHA codes

df = px.data.gapminder().query("year==2007")[["country", "iso_alpha"]]



# Merge the tables together (we lose a fiew rows, but not many)

data = pd.merge(left=train, right=df, how="inner", on="country")



# Group by country and count how many species can be found in each

data = data.groupby(by=["country", "iso_alpha"]).count()["species"].reset_index()



fig = px.choropleth(data, locations="iso_alpha", color="species", hover_name="country",

                    color_continuous_scale=px.colors.sequential.Teal,

                    title = "World Map: Recordings per Country")

fig.show()
#Total no of unique species in the dataset

display(len(train['species'].value_counts().index))
train['species'].value_counts()
top15_speies=list(train['country'].value_counts().head(10).reset_index()['index'])

top10

# SHP file

world_map = gpd.read_file("../input/world-shapefile/world_shapefile.shp")



# Coordinate reference system

crs = {"init" : "epsg:4326"}



# Lat and Long need to be of type float, not object

data = train[train["latitude"] != "Not specified"]

data["latitude"] = data["latitude"].astype(float)

data["longitude"] = data["longitude"].astype(float)



# Create geometry

geometry = [Point(xy) for xy in zip(data["longitude"], data["latitude"])]



# Geo Dataframe

geo_df = gpd.GeoDataFrame(data, crs=crs, geometry=geometry)



# Create ID for species

species_id = geo_df["species"].value_counts().reset_index()

species_id.insert(0, 'ID', range(0, 0 + len(species_id)))



species_id.columns = ["ID", "species", "count"]



# Add ID to geo_df

geo_df = pd.merge(geo_df, species_id, how="left", on="species")



# === PLOT ===

fig, ax = plt.subplots(figsize = (16, 10))

world_map.plot(ax=ax, alpha=0.4, color="grey")



palette = iter(sns.hls_palette(len(species_id)))



for i in range(264):

    geo_df[geo_df["ID"] == i].plot(ax=ax, markersize=20, color=next(palette), marker="*", label = "test");

train['full_path']=audio+train['ebird_code']+'/'+train['filename']

# Now let's sample a few audio files

# Now let's sample a fiew audio files

haiwoo = train[train['ebird_code'] == "haiwoo"].sample(1, random_state = 33)['full_path'].values[0]

wesmea = train[train['ebird_code'] == "wesmea"].sample(1, random_state = 33)['full_path'].values[0]

wewpew = train[train['ebird_code'] == "wewpew"].sample(1, random_state = 33)['full_path'].values[0]

scoori = train[train['ebird_code'] == "scoori"].sample(1, random_state = 33)['full_path'].values[0]

bewwre = train[train['ebird_code'] == "bewwre"].sample(1, random_state = 33)['full_path'].values[0]



bird_sample_list = ["haiwoo", "wesmea", "wewpew", "scoori", "bewwre"]
#bewwre

ipd.Audio(bewwre)
#wesmea

ipd.Audio(wesmea)
#wewpew

ipd.Audio(wewpew)
#scoori

ipd.Audio(scoori)
#bewwre

ipd.Audio(bewwre)
# Importing 1 file

y, sr = librosa.load(bewwre)



print('y:', y, '\n')

print('y shape:', np.shape(y), '\n')

print('Sample Rate (KHz):', sr, '\n')



# Verify length of the audio

print('Check Len of Audio:', 661794/sr)
# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)

audio_file, _ = librosa.effects.trim(y)



# the result is an numpy ndarray

print('Audio File:', audio_file, '\n')

print('Audio File shape:', np.shape(audio_file))
#Importing the 5 files

y_haiwoo,sr_haiwoo=librosa.load(haiwoo)

audio_haiwoo,_=librosa.effects.trim(y_haiwoo)

y_wesmea,sr_wesmea=librosa.load(wesmea)

audio_wesmea,_=librosa.effects.trim(y_wesmea)

y_wewpew,sr_wewpew=librosa.load(wewpew)

audio_wewpew,_=librosa.effects.trim(y_wewpew)

y_scoori,sr_scoori=librosa.load(scoori)

audio_scoori,_=librosa.effects.trim(y_scoori)

y_bewwre,sr_bewwre=librosa.load(bewwre)

audio_bewwre,_=librosa.effects.trim(y_bewwre)
fig, ax = plt.subplots(5, figsize = (16, 9))

fig.suptitle('Sound Waves', fontsize=16)



librosa.display.waveplot(y = audio_haiwoo, sr =sr_haiwoo, color = "#A300F9", ax=ax[0])

librosa.display.waveplot(y = audio_wesmea, sr = sr_wesmea, color = "#4300FF", ax=ax[1])

librosa.display.waveplot(y = audio_wewpew, sr = sr_wewpew, color = "#009DFF", ax=ax[2])

librosa.display.waveplot(y = audio_scoori, sr = sr_scoori, color = "#00FFB0", ax=ax[3])

librosa.display.waveplot(y = audio_bewwre, sr = sr_bewwre, color = "#D9FF00", ax=ax[4]);



for i, name in zip(range(5), bird_sample_list):

    ax[i].set_ylabel(name, fontsize=13)
# Default FFT window size

n_fft = 2048 # FFT window size

hop_length = 512 # number audio of frames between STFT columns (looks like a good default)



# Short-time Fourier transform (STFT)

D_haiwoo = np.abs(librosa.stft(audio_haiwoo, n_fft = n_fft, hop_length = hop_length))

D_wesmea = np.abs(librosa.stft(audio_wesmea, n_fft = n_fft, hop_length = hop_length))

D_wewpew = np.abs(librosa.stft(audio_wewpew, n_fft = n_fft, hop_length = hop_length))

D_scoori = np.abs(librosa.stft(audio_scoori, n_fft = n_fft, hop_length = hop_length))

D_bewwre = np.abs(librosa.stft(audio_bewwre, n_fft = n_fft, hop_length = hop_length))
D_birds_list = [D_haiwoo, D_wesmea, D_wewpew,D_scoori, D_bewwre]



for bird,name in zip(D_birds_list,bird_sample_list):

    print(" shape is",name,np.shape(bird))
# Convert an amplitude spectrogram to Decibels-scaled spectrogram.

DB_haiwoo = librosa.amplitude_to_db(D_haiwoo, ref = np.max)

DB_wesmea = librosa.amplitude_to_db(D_wesmea, ref = np.max)

DB_wewpew = librosa.amplitude_to_db(D_wewpew, ref = np.max)

DB_scoori = librosa.amplitude_to_db(D_scoori, ref = np.max)

DB_bewwre = librosa.amplitude_to_db(D_bewwre, ref = np.max)



# === PLOT ===

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

fig.suptitle('Spectrogram', fontsize=16)

fig.delaxes(ax[1, 2])



librosa.display.specshow(DB_haiwoo, sr = sr_haiwoo, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[0, 0])



librosa.display.specshow(DB_wesmea, sr = sr_wesmea, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[0, 1])



librosa.display.specshow(DB_wewpew, sr = sr_wewpew, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[0, 2])

librosa.display.specshow(DB_scoori, sr = sr_scoori, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[1, 0])



librosa.display.specshow(DB_bewwre, sr = sr_bewwre, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[1, 1]);







for i, name in zip(range(0, 2*3), bird_sample_list):

    x = i // 3

    y = i % 3

    ax[x, y].set_title(name, fontsize=13) 



# === PLOT ===

fig, ax = plt.subplots(2, 3, figsize=(17, 5))

fig.suptitle('Waveform', fontsize=16)

fig.delaxes(ax[1, 2])



librosa.display.waveplot(DB_haiwoo, sr = sr_haiwoo, x_axis = 'time', 

                          ax=ax[0, 0])

librosa.display.waveplot(DB_wesmea, sr = sr_wesmea, x_axis = 'time', 

                          ax=ax[0, 1])

librosa.display.waveplot(DB_wewpew, sr = sr_wewpew, x_axis = 'time', 

                          ax=ax[0, 2])

librosa.display.waveplot(DB_scoori, sr = sr_scoori, x_axis = 'time', 

                          ax=ax[1, 0])

librosa.display.waveplot(DB_bewwre, sr = sr_bewwre, x_axis = 'time', 

                         ax=ax[1, 1]);

for i, name in zip(range(0,2*3), bird_sample_list):

    x = i // 3

    y = i % 3

    ax[x, y].set_title(name, fontsize=13) 



# Total zero_crossings in our 1 song

zero_haiwoo = librosa.zero_crossings(audio_haiwoo, pad=False)

zero_wesmea= librosa.zero_crossings(audio_wesmea, pad=False)

zero_wewpew = librosa.zero_crossings(audio_wewpew, pad=False)

zero_scoori = librosa.zero_crossings(audio_scoori, pad=False)

zero_bewwre = librosa.zero_crossings(audio_bewwre, pad=False)



zero_birds_list = [zero_haiwoo, zero_wesmea, zero_wewpew,zero_scoori, zero_bewwre]



for bird, name in zip(zero_birds_list, bird_sample_list):

    print("{} change rate is {:,}".format(name, sum(bird)))
y_harm_haiwoo, y_perc_haiwoo = librosa.effects.hpss(audio_haiwoo)

y_harm_wesmea, y_perc_wesmea = librosa.effects.hpss(audio_wesmea)

y_harm_wewpew, y_perc_wewpew = librosa.effects.hpss(audio_wewpew)

y_harm_scoori, y_perc_scoori = librosa.effects.hpss(audio_scoori)

y_harm_bewwre, y_perc_bewwre = librosa.effects.hpss(audio_bewwre)

#for haiwoo

plt.figure(figsize = (16, 6))

plt.plot(y_perc_haiwoo, color = '#FFB100')

plt.plot(y_harm_haiwoo, color = '#A300F9')

plt.legend(("Perceptrual", "Harmonics"))

plt.title("Harmonics and Perceptrual : Haiwoo Bird", fontsize=16);

#for wesmea

plt.figure(figsize = (16, 6))

plt.plot(y_perc_wesmea, color = '#FFB150')

plt.plot(y_harm_wesmea, color = '#A300F9')

plt.legend(("Perceptrual", "Harmonics"))

plt.title("Harmonics and Perceptrual : wesmea Bird", fontsize=16);

#for wewpew

plt.figure(figsize = (16, 6))

plt.plot(y_perc_wewpew, color = '#FFB234')

plt.plot(y_harm_wewpew, color = '#A300F9')

plt.legend(("Perceptrual", "Harmonics"))

plt.title("Harmonics and Perceptrual : wewpew Bird", fontsize=16);

#for scoori

plt.figure(figsize = (16, 6))

plt.plot(y_perc_scoori, color = '#FFBB10')

plt.plot(y_harm_scoori, color = '#A300F9')

plt.legend(("Perceptrual", "Harmonics"))

plt.title("Harmonics and Perceptrual : Scoori Bird", fontsize=16);

#for bewwre

plt.figure(figsize = (16, 6))

plt.plot(y_perc_bewwre, color = '#FFB100')

plt.plot(y_harm_bewwre, color = '#A300F9')

plt.legend(("Perceptrual", "Harmonics"))

plt.title("Harmonics and Perceptrual : bewwre Bird", fontsize=16);
# Calculate the Spectral Centroids

spectral_centroids_scoori = librosa.feature.spectral_centroid(audio_scoori, sr=sr_scoori)[0]

spectral_centroids_haiwoo = librosa.feature.spectral_centroid(audio_haiwoo, sr=sr_haiwoo)[0]

spectral_centroids_wewpew = librosa.feature.spectral_centroid(audio_wewpew, sr=sr_wewpew)[0]

spectral_centroids_wesmea = librosa.feature.spectral_centroid(audio_wesmea, sr=sr_wesmea)[0]

spectral_centroids_bewwre = librosa.feature.spectral_centroid(audio_bewwre, sr=sr_bewwre)[0]



spectral_centroids=[spectral_centroids_haiwoo,spectral_centroids_wewpew,spectral_centroids_wesmea,spectral_centroids_bewwre,spectral_centroids_scoori]





# Shape is a vector

for Centroids, name in zip(spectral_centroids, bird_sample_list):

    print("Centroids of ",name,'is',Centroids,'\n')



# Computing the time variable for visualization

frames = range(len(Centroids))



# Converts frame counts to time (seconds)

t = librosa.frames_to_time(frames)



print('frames:', frames, '\n')

print('t:', t)



# Function that normalizes the Sound Data

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform

#for Haiwoo

plt.figure(figsize = (19, 6))

librosa.display.waveplot(audio_haiwoo, sr=sr_haiwoo, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Centroids), color='#FFB100', lw=2)

plt.legend(["Spectral Centroid", "Wave"])

plt.title("Spectral Centroid: Haiwoo Bird", fontsize=16);

#for Wewpew

plt.figure(figsize=(19,6))

librosa.display.waveplot(audio_wewpew, sr=sr_wewpew, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Centroids), color='#FFB100', lw=2)

plt.legend(["Spectral Centroid", "Wave"])

plt.title("Spectral Centroid:Wewpew Bird", fontsize=16);

#for Wesmea

plt.figure(figsize=(19,6))

librosa.display.waveplot(audio_wesmea, sr=sr_wesmea, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Centroids), color='#FFB100', lw=2)

plt.legend(["Spectral Centroid", "Wave"])

plt.title("Spectral Centroid: Wesmea Bird", fontsize=16);

# for bewwre

plt.figure(figsize=(19,6))

librosa.display.waveplot(audio_bewwre, sr=sr_bewwre, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Centroids), color='#FFB100', lw=2)

plt.legend(["Spectral Centroid", "Wave"])

plt.title("Spectral Centroid: Bewwre Bird", fontsize=16);

#for Scoori

plt.figure(figsize=(19,6))

librosa.display.waveplot(audio_scoori, sr=sr_scoori, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Centroids), color='#FFB100', lw=2)

plt.legend(["Spectral Centroid", "Wave"])

plt.title("Spectral Centroid: Scoori Bird", fontsize=16)



# Increase or decrease hop_length to change how granular you want your data to be

hop_length = 5000



# Chromogram Vesspa

chromagram_haiwoo = librosa.feature.chroma_stft(audio_haiwoo, sr=sr_haiwoo, hop_length=hop_length)

chromagram_wewpew = librosa.feature.chroma_stft(audio_wewpew, sr=sr_wewpew, hop_length=hop_length)

chromagram_wesmea = librosa.feature.chroma_stft(audio_wesmea, sr=sr_wesmea, hop_length=hop_length)

chromagram_bewwre = librosa.feature.chroma_stft(audio_bewwre, sr=sr_bewwre, hop_length=hop_length)

chromagram_scoori = librosa.feature.chroma_stft(audio_scoori, sr=sr_scoori, hop_length=hop_length)

chromagram=[chromagram_haiwoo,chromagram_wewpew,chromagram_wesmea,chromagram_bewwre,chromagram_scoori]

for bird,name in zip(chromagram,bird_sample_list):

    print("Chromogram",name,'shape',np.shape(bird))



#for Haiwoo

librosa.display.specshow(chromagram_haiwoo, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')

plt.figure(figsize=(16, 6))

plt.title("Chromogram Haiwoo", fontsize=16);

#for wewpew

librosa.display.specshow(chromagram_wewpew, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')

plt.figure(figsize=(16, 6))

plt.title("Chromogram Wewpew", fontsize=16);

#for Wesmea

librosa.display.specshow(chromagram_wesmea, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')

plt.figure(figsize=(16, 6))

plt.title("Chromogram Wesmea", fontsize=16);

#for Bewwre

librosa.display.specshow(chromagram_bewwre, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')

plt.figure(figsize=(16, 6))

plt.title("Chromogram Bewwre", fontsize=16);

#for scoori

librosa.display.specshow(chromagram_scoori, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')

plt.figure(figsize=(16, 6))

plt.title("Chromogram Scoori", fontsize=16);

#

# Spectral RollOff Vector

spectral_rolloff_haiwoo = librosa.feature.spectral_rolloff(audio_haiwoo, sr=sr_haiwoo)[0]

spectral_rolloff_wewpew= librosa.feature.spectral_rolloff(audio_wewpew, sr=sr_wewpew)[0]

spectral_rolloff_wesmea= librosa.feature.spectral_rolloff(audio_wesmea, sr=sr_wesmea)[0]

spectral_rolloff_bewwre= librosa.feature.spectral_rolloff(audio_bewwre, sr=sr_bewwre)[0]

spectral_rolloff_scoori= librosa.feature.spectral_rolloff(audio_scoori, sr=sr_scoori)[0]



spectral_rolloff=[spectral_rolloff_haiwoo,spectral_rolloff_wewpew,spectral_rolloff_wesmea,spectral_rolloff_bewwre,spectral_rolloff_scoori]

# Shape is a vector

for Rolloff, name in zip(spectral_rolloff, bird_sample_list):

    print("Centroids of ",name,'is',Rolloff,'\n')



# Computing the time variable for visualization

frames = range(len(Rolloff))

# Converts frame counts to time (seconds)

t = librosa.frames_to_time(frames)

print('frames:', frames, '\n')

print('t:', t)





# The plot

#for Haiwoo

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_haiwoo, sr=sr_haiwoo, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: haiwoo Bird", fontsize=16);

#for Wesmea

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_wesmea, sr=sr_wesmea, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: Wesmea Bird", fontsize=16);\

# for wewpew

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_wewpew, sr=sr_wewpew, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: Wewpew Bird", fontsize=16);

#for bewwri

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_bewwre, sr=sr_bewwre, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: Bewwre Bird", fontsize=16);

#for Scoori

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_scoori, sr=sr_scoori, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(Rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: Scoori Bird", fontsize=16);