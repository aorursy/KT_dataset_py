import os



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

from matplotlib.offsetbox import AnnotationBbox, OffsetImage



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
# Import data

train_csv = pd.read_csv("../input/birdsong-recognition/train.csv")

test_csv = pd.read_csv("../input/birdsong-recognition/test.csv")



# Create some time features

train_csv['year'] = train_csv['date'].apply(lambda x: x.split('-')[0])

train_csv['month'] = train_csv['date'].apply(lambda x: x.split('-')[1])

train_csv['day_of_month'] = train_csv['date'].apply(lambda x: x.split('-')[2])



print("There are {:,} unique bird species in the dataset.".format(len(train_csv['species'].unique())))
# Inspect text_csv before checking train data

test_csv
bird = mpimg.imread('../input/birdcall-recognition-data/pink bird.jpg')

imagebox = OffsetImage(bird, zoom=0.5)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(6.5, 2000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['year'], palette="hls")

ax.add_artist(ab)



plt.title("Audio Files Registration per Year Made", fontsize=16)

plt.xticks(rotation=90, fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("");
bird = mpimg.imread('../input/birdcall-recognition-data/green bird.jpg')

imagebox = OffsetImage(bird, zoom=0.3)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(11, 3000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['month'], palette="hls")

ax.add_artist(ab)



plt.title("Audio Files Registration per Month Made", fontsize=16)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("");
bird = mpimg.imread('../input/birdcall-recognition-data/orangebird.jpeg')

imagebox = OffsetImage(bird, zoom=0.12)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(3.9, 8600))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['pitch'], palette="hls", order = train_csv['pitch'].value_counts().index)

ax.add_artist(ab)



plt.title("Pitch (quality of sound - how high/low the tone is)", fontsize=16)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("");
# Create a new variable type by exploding all the values

adjusted_type = train_csv['type'].apply(lambda x: x.split(',')).reset_index().explode("type")



# Strip of white spaces and convert to lower chars

adjusted_type = adjusted_type['type'].apply(lambda x: x.strip().lower()).reset_index()

adjusted_type['type'] = adjusted_type['type'].replace('calls', 'call')



# Create Top 15 list with song types

top_15 = list(adjusted_type['type'].value_counts().head(15).reset_index()['index'])

data = adjusted_type[adjusted_type['type'].isin(top_15)]



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/Eastern Meadowlark.jpg')

imagebox = OffsetImage(bird, zoom=0.43)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(12.4, 5700))



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['type'], palette="hls", order = data['type'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 Song Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Top 15 most common elevations

top_15 = list(train_csv['elevation'].value_counts().head(15).reset_index()['index'])

data = train_csv[train_csv['elevation'].isin(top_15)]



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/blue bird.jpg')

imagebox = OffsetImage(bird, zoom=0.43)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(12.4, 1450))



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['elevation'], palette="hls", order = data['elevation'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 Elevation Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Create data

data = train_csv['bird_seen'].value_counts().reset_index()



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/black bird.jpg')

imagebox = OffsetImage(bird, zoom=0.22)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(15300, 0.95))



plt.figure(figsize=(16, 6))

ax = sns.barplot(x = 'bird_seen', y = 'index', data = data, palette="hls")

ax.add_artist(ab)



plt.title("Song was Heard, but was Bird Seen?", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Top 15 most common elevations

top_15 = list(train_csv['country'].value_counts().head(15).reset_index()['index'])

data = train_csv[train_csv['country'].isin(top_15)]



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/fluff ball.jpg')

imagebox = OffsetImage(bird, zoom=0.6)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(12.2, 7000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['country'], palette='hls', order = data['country'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 Countries with most Recordings", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Import gapminder data, where we have country and iso ALPHA codes

df = px.data.gapminder().query("year==2007")[["country", "iso_alpha"]]



# Merge the tables together (we lose a fiew rows, but not many)

data = pd.merge(left=train_csv, right=df, how="inner", on="country")



# Group by country and count how many species can be found in each

data = data.groupby(by=["country", "iso_alpha"]).count()["species"].reset_index()



fig = px.choropleth(data, locations="iso_alpha", color="species", hover_name="country",

                    color_continuous_scale=px.colors.sequential.Teal,

                    title = "World Map: Recordings per Country")

fig.show()
# SHP file

world_map = gpd.read_file("../input/world-shapefile/world_shapefile.shp")



# Coordinate reference system

crs = {"init" : "epsg:4326"}



# Lat and Long need to be of type float, not object

data = train_csv[train_csv["latitude"] != "Not specified"]

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

    geo_df[geo_df["ID"] == i].plot(ax=ax, markersize=20, color=next(palette), marker="o", label = "test");
# Creating Interval for *duration* variable

train_csv['duration_interval'] = ">500"

train_csv.loc[train_csv['duration'] <= 100, 'duration_interval'] = "<=100"

train_csv.loc[(train_csv['duration'] > 100) & (train_csv['duration'] <= 200), 'duration_interval'] = "100-200"

train_csv.loc[(train_csv['duration'] > 200) & (train_csv['duration'] <= 300), 'duration_interval'] = "200-300"

train_csv.loc[(train_csv['duration'] > 300) & (train_csv['duration'] <= 400), 'duration_interval'] = "300-400"

train_csv.loc[(train_csv['duration'] > 400) & (train_csv['duration'] <= 500), 'duration_interval'] = "400-500"



bird = mpimg.imread('../input/birdcall-recognition-data/multicolor bird.jpg')

imagebox = OffsetImage(bird, zoom=0.4)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(4.4, 12000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['duration_interval'], palette="hls")

ax.add_artist(ab)



plt.title("Distribution of Recordings Duration", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
def show_values_on_bars(axs, h_v="v", space=0.4):

    def _show_on_single_plot(ax):

        if h_v == "v":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() / 2

                _y = p.get_y() + p.get_height()

                value = int(p.get_height())

                ax.text(_x, _y, value, ha="center") 

        elif h_v == "h":

            for p in ax.patches:

                _x = p.get_x() + p.get_width() + float(space)

                _y = p.get_y() + p.get_height()

                value = int(p.get_width())

                ax.text(_x, _y, value, ha="left")



    if isinstance(axs, np.ndarray):

        for idx, ax in np.ndenumerate(axs):

            _show_on_single_plot(ax)

    else:

        _show_on_single_plot(axs)
bird = mpimg.imread('../input/birdcall-recognition-data/yellow birds.jpg')

imagebox = OffsetImage(bird, zoom=0.6)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(2.7, 12000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['file_type'], palette = "hls", order = train_csv['file_type'].value_counts().index)

ax.add_artist(ab)



show_values_on_bars(ax, "v", 0)



plt.title("Recording File Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Create Full Path so we can access data more easily

base_dir = '../input/birdsong-recognition/train_audio/'

train_csv['full_path'] = base_dir + train_csv['ebird_code'] + '/' + train_csv['filename']



# Now let's sample a fiew audio files

amered = train_csv[train_csv['ebird_code'] == "amered"].sample(1, random_state = 33)['full_path'].values[0]

cangoo = train_csv[train_csv['ebird_code'] == "cangoo"].sample(1, random_state = 33)['full_path'].values[0]

haiwoo = train_csv[train_csv['ebird_code'] == "haiwoo"].sample(1, random_state = 33)['full_path'].values[0]

pingro = train_csv[train_csv['ebird_code'] == "pingro"].sample(1, random_state = 33)['full_path'].values[0]

vesspa = train_csv[train_csv['ebird_code'] == "vesspa"].sample(1, random_state = 33)['full_path'].values[0]



bird_sample_list = ["amered", "cangoo", "haiwoo", "pingro", "vesspa"]
# Amered

ipd.Audio(amered)
# Cangoo

ipd.Audio(cangoo)
# Haiwoo

ipd.Audio(haiwoo)
# Pingro

ipd.Audio(pingro)
# Vesspa

ipd.Audio(vesspa)
# Importing 1 file

y, sr = librosa.load(vesspa)



print('y:', y, '\n')

print('y shape:', np.shape(y), '\n')

print('Sample Rate (KHz):', sr, '\n')



# Verify length of the audio

print('Check Len of Audio:', np.shape(y)[0]/sr)
# Trim leading and trailing silence from an audio signal (silence before and after the actual audio)

audio_file, _ = librosa.effects.trim(y)



# the result is an numpy ndarray

print('Audio File:', audio_file, '\n')

print('Audio File shape:', np.shape(audio_file))
# Importing the 5 files

y_amered, sr_amered = librosa.load(amered)

audio_amered, _ = librosa.effects.trim(y_amered)



y_cangoo, sr_cangoo = librosa.load(cangoo)

audio_cangoo, _ = librosa.effects.trim(y_cangoo)



y_haiwoo, sr_haiwoo = librosa.load(haiwoo)

audio_haiwoo, _ = librosa.effects.trim(y_haiwoo)



y_pingro, sr_pingro = librosa.load(pingro)

audio_pingro, _ = librosa.effects.trim(y_pingro)



y_vesspa, sr_vesspa = librosa.load(vesspa)

audio_vesspa, _ = librosa.effects.trim(y_vesspa)
fig, ax = plt.subplots(5, figsize = (16, 9))

fig.suptitle('Sound Waves', fontsize=16)



librosa.display.waveplot(y = audio_amered, sr = sr_amered, color = "#A300F9", ax=ax[0])

librosa.display.waveplot(y = audio_cangoo, sr = sr_cangoo, color = "#4300FF", ax=ax[1])

librosa.display.waveplot(y = audio_haiwoo, sr = sr_haiwoo, color = "#009DFF", ax=ax[2])

librosa.display.waveplot(y = audio_pingro, sr = sr_pingro, color = "#00FFB0", ax=ax[3])

librosa.display.waveplot(y = audio_vesspa, sr = sr_vesspa, color = "#D9FF00", ax=ax[4]);



for i, name in zip(range(5), bird_sample_list):

    ax[i].set_ylabel(name, fontsize=13)
# Default FFT window size

n_fft = 2048 # FFT window size

hop_length = 512 # number audio of frames between STFT columns (looks like a good default)



# Short-time Fourier transform (STFT)

D_amered = np.abs(librosa.stft(audio_amered, n_fft = n_fft, hop_length = hop_length))

D_cangoo = np.abs(librosa.stft(audio_cangoo, n_fft = n_fft, hop_length = hop_length))

D_haiwoo = np.abs(librosa.stft(audio_haiwoo, n_fft = n_fft, hop_length = hop_length))

D_pingro = np.abs(librosa.stft(audio_pingro, n_fft = n_fft, hop_length = hop_length))

D_vesspa = np.abs(librosa.stft(audio_vesspa, n_fft = n_fft, hop_length = hop_length))
print('Shape of D object:', np.shape(D_amered))
# Convert an amplitude spectrogram to Decibels-scaled spectrogram.

DB_amered = librosa.amplitude_to_db(D_amered, ref = np.max)

DB_cangoo = librosa.amplitude_to_db(D_cangoo, ref = np.max)

DB_haiwoo = librosa.amplitude_to_db(D_haiwoo, ref = np.max)

DB_pingro = librosa.amplitude_to_db(D_pingro, ref = np.max)

DB_vesspa = librosa.amplitude_to_db(D_vesspa, ref = np.max)



# === PLOT ===

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

fig.suptitle('Spectrogram', fontsize=16)

fig.delaxes(ax[1, 2])



librosa.display.specshow(DB_amered, sr = sr_amered, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[0, 0])

librosa.display.specshow(DB_cangoo, sr = sr_cangoo, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[0, 1])

librosa.display.specshow(DB_haiwoo, sr = sr_haiwoo, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[0, 2])

librosa.display.specshow(DB_pingro, sr = sr_pingro, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[1, 0])

librosa.display.specshow(DB_vesspa, sr = sr_vesspa, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'cool', ax=ax[1, 1]);



for i, name in zip(range(0, 2*3), bird_sample_list):

    x = i // 3

    y = i % 3

    ax[x, y].set_title(name, fontsize=13) 
# Create the Mel Spectrograms

S_amered = librosa.feature.melspectrogram(y_amered, sr=sr_amered)

S_DB_amered = librosa.amplitude_to_db(S_amered, ref=np.max)



S_cangoo = librosa.feature.melspectrogram(y_cangoo, sr=sr_cangoo)

S_DB_cangoo = librosa.amplitude_to_db(S_cangoo, ref=np.max)



S_haiwoo = librosa.feature.melspectrogram(y_haiwoo, sr=sr_haiwoo)

S_DB_haiwoo = librosa.amplitude_to_db(S_haiwoo, ref=np.max)



S_pingro = librosa.feature.melspectrogram(y_pingro, sr=sr_pingro)

S_DB_pingro = librosa.amplitude_to_db(S_pingro, ref=np.max)



S_vesspa = librosa.feature.melspectrogram(y_vesspa, sr=sr_vesspa)

S_DB_vesspa = librosa.amplitude_to_db(S_vesspa, ref=np.max)



# === PLOT ====

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

fig.suptitle('Mel Spectrogram', fontsize=16)

fig.delaxes(ax[1, 2])



librosa.display.specshow(S_DB_amered, sr = sr_amered, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'rainbow', ax=ax[0, 0])

librosa.display.specshow(S_DB_cangoo, sr = sr_cangoo, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'rainbow', ax=ax[0, 1])

librosa.display.specshow(S_DB_haiwoo, sr = sr_haiwoo, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'rainbow', ax=ax[0, 2])

librosa.display.specshow(S_DB_pingro, sr = sr_pingro, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'rainbow', ax=ax[1, 0])

librosa.display.specshow(S_DB_vesspa, sr = sr_vesspa, hop_length = hop_length, x_axis = 'time', 

                         y_axis = 'log', cmap = 'rainbow', ax=ax[1, 1]);



for i, name in zip(range(0, 2*3), bird_sample_list):

    x = i // 3

    y = i % 3

    ax[x, y].set_title(name, fontsize=13)
# Total zero_crossings in our 1 song

zero_amered = librosa.zero_crossings(audio_amered, pad=False)

zero_cangoo = librosa.zero_crossings(audio_cangoo, pad=False)

zero_haiwoo = librosa.zero_crossings(audio_haiwoo, pad=False)

zero_pingro = librosa.zero_crossings(audio_pingro, pad=False)

zero_vesspa = librosa.zero_crossings(audio_vesspa, pad=False)



zero_birds_list = [zero_amered, zero_cangoo, zero_haiwoo, zero_pingro, zero_vesspa]



for bird, name in zip(zero_birds_list, bird_sample_list):

    print("{} change rate is {:,}".format(name, sum(bird)))
y_harm_haiwoo, y_perc_haiwoo = librosa.effects.hpss(audio_haiwoo)



plt.figure(figsize = (16, 6))

plt.plot(y_perc_haiwoo, color = '#FFB100')

plt.plot(y_harm_haiwoo, color = '#A300F9')

plt.legend(("Perceptrual", "Harmonics"))

plt.title("Harmonics and Perceptrual : Haiwoo Bird", fontsize=16);
# Calculate the Spectral Centroids

spectral_centroids = librosa.feature.spectral_centroid(audio_cangoo, sr=sr)[0]



# Shape is a vector

print('Centroids:', spectral_centroids, '\n')

print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')



# Computing the time variable for visualization

frames = range(len(spectral_centroids))



# Converts frame counts to time (seconds)

t = librosa.frames_to_time(frames)



print('frames:', frames, '\n')

print('t:', t)



# Function that normalizes the Sound Data

def normalize(x, axis=0):

    return sklearn.preprocessing.minmax_scale(x, axis=axis)
#Plotting the Spectral Centroid along the waveform

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_cangoo, sr=sr, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(spectral_centroids), color='#FFB100', lw=2)

plt.legend(["Spectral Centroid", "Wave"])

plt.title("Spectral Centroid: Cangoo Bird", fontsize=16);
# Increase or decrease hop_length to change how granular you want your data to be

hop_length = 5000



# Chromogram Vesspa

chromagram = librosa.feature.chroma_stft(audio_vesspa, sr=sr_vesspa, hop_length=hop_length)

print('Chromogram Vesspa shape:', chromagram.shape)



plt.figure(figsize=(16, 6))

librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='twilight')



plt.title("Chromogram: Vesspa", fontsize=16);
# Create Tempo BPM variable

tempo_amered, _ = librosa.beat.beat_track(y_amered, sr = sr_amered)

tempo_cangoo, _ = librosa.beat.beat_track(y_cangoo, sr = sr_cangoo)

tempo_haiwoo, _ = librosa.beat.beat_track(y_haiwoo, sr = sr_haiwoo)

tempo_pingro, _ = librosa.beat.beat_track(y_pingro, sr = sr_pingro)

tempo_vesspa, _ = librosa.beat.beat_track(y_vesspa, sr = sr_vesspa)



data = pd.DataFrame({"Type": bird_sample_list , 

                     "BPM": [tempo_amered, tempo_cangoo, tempo_haiwoo, tempo_pingro, tempo_vesspa] })



# Image

bird = mpimg.imread('../input/birdcall-recognition-data/violet bird.jpg')

imagebox = OffsetImage(bird, zoom=0.34)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(0.05, 158))



# Plot

plt.figure(figsize = (16, 6))

ax = sns.barplot(y = data["BPM"], x = data["Type"], palette="hls")

ax.add_artist(ab)



plt.ylabel("BPM", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(fontsize=13)

plt.xlabel("")

plt.title("BPM for 5 Different Bird Species", fontsize=16);
# Spectral RollOff Vector

spectral_rolloff = librosa.feature.spectral_rolloff(audio_amered, sr=sr_amered)[0]



# Computing the time variable for visualization

frames = range(len(spectral_rolloff))

# Converts frame counts to time (seconds)

t = librosa.frames_to_time(frames)



# The plot

plt.figure(figsize = (16, 6))

librosa.display.waveplot(audio_amered, sr=sr_amered, alpha=0.4, color = '#A300F9', lw=3)

plt.plot(t, normalize(spectral_rolloff), color='#FFB100', lw=3)

plt.legend(["Spectral Rolloff", "Wave"])

plt.title("Spectral Rolloff: Amered Bird", fontsize=16);
# Import the .csv files (corresponding with the extended data)

train_extended_A_Z = pd.read_csv("../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv")



# Create base directory

base_dir_A_M = "../input/xeno-canto-bird-recordings-extended-a-m"

base_dir_N_Z = "../input/xeno-canto-bird-recordings-extended-n-z"



# Create Full Path column to the audio files

train_extended_A_Z['full_path'] = base_dir_A_M + train_extended_A_Z['ebird_code'] + '/' + train_extended_A_Z['filename']
def count_files_dir(dir_name = "Default", pref = "Def"):

    

    birds_names = list(os.listdir(dir_name + "/" + pref))

    total_len = 0



    for bird in birds_names:

        total_len += len(os.listdir(dir_name +"/" + pref + "/" + bird))

        

    return total_len
A_M = count_files_dir(base_dir_A_M, pref = "A-M")

N_Z = count_files_dir(base_dir_N_Z, pref = "N-Z")



print("There are {:,} birds in A-Z .csv file".format(len(train_extended_A_Z)), "\n" +

      "\n" +

      "and there are {:,} audio recs.".format(A_M + N_Z))