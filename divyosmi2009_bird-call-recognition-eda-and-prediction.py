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
train_csv = pd.read_csv("../input/birdsong-recognition/train.csv")

test_csv = pd.read_csv("../input/birdsong-recognition/test.csv")

# Create some time features

train_csv['year'] = train_csv['date'].apply(lambda x: x.split('-')[0])

train_csv['month'] = train_csv['date'].apply(lambda x: x.split('-')[1])

train_csv['day_of_month'] = train_csv['date'].apply(lambda x: x.split('-')[2])

print(train_csv.describe())

print(len(train_csv))

print("nan or null values {}".format(train_csv.columns[train_csv.isna().any()]))

print("There are {:,} unique bird species in the dataset.".format(len(train_csv['species'].unique())))
for i in train_csv.columns:

    print(i)
bird = mpimg.imread('../input/birdcall-recognition-data/pink bird.jpg')

imagebox = OffsetImage(bird, zoom=0.5)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(6.5, 2000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['year'], palette="rainbow")

ax.add_artist(ab)



plt.title("Audio Files Registration per Year Made", fontsize=16)

plt.xticks(rotation=90, fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("year");
bird = mpimg.imread('../input/birdcall-recognition-data/blue bird.jpg')

imagebox = OffsetImage(bird, zoom=0.3)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(10, 3000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['month'], palette="winter")

ax.add_artist(ab)



plt.title("Audio Files Registration per Month Made", fontsize=16)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("month")
bird = mpimg.imread('../input/birdcall-recognition-data/Eastern Meadowlark.jpg')

imagebox = OffsetImage(bird, zoom=0.30)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(3.9, 8600))



plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['pitch'], palette="Reds", order = train_csv['pitch'].value_counts().index)

ax.add_artist(ab)



plt.title("Pitch (quality of sound - how high/low the tone is)", fontsize=16)

plt.xticks(fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Frequency", fontsize=14)

plt.xlabel("")
adjusted_type = train_csv['type'].apply(lambda x: x.split(',')).reset_index().explode("type")



# Strip of white spaces and convert to lower chars

adjusted_type = adjusted_type['type'].apply(lambda x: x.strip().lower()).reset_index()

adjusted_type['type'] = adjusted_type['type'].replace('calls', 'call')



# Create Top 15 list with song types

top_15 = list(adjusted_type['type'].value_counts().head(15).reset_index()['index'])

data = adjusted_type[adjusted_type['type'].isin(top_15)]



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/multicolor bird.jpg')

imagebox = OffsetImage(bird, zoom=0.43)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(11, 5700))



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['type'], palette="Greens", order = data['type'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 Song Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("songs");
top_15 = list(train_csv['elevation'].value_counts().head(15).reset_index()['index'])

data = train_csv[train_csv['elevation'].isin(top_15)]



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/violet bird.jpg')

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

plt.xlabel("Elevation(in metre / m)")
data = train_csv['bird_seen'].value_counts().reset_index()



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/black bird.jpg')

imagebox = OffsetImage(bird, zoom=0.22)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(15300, 0.95))



plt.figure(figsize=(16, 6))

ax = sns.barplot(x = 'bird_seen', y = 'index', data = data, palette="hls")

ax.add_artist(ab)



plt.title("heard song what about the bird?", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("per survey");
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

plt.xlabel("countries")
top_15 = list(train_csv['length'].value_counts().head(15).reset_index()['index'])

data = train_csv[train_csv['length'].isin(top_15)]



# === PLOT ===

plt.figure(figsize=(16, 6))

ax = sns.countplot(data['length'], palette="hls", order = data['length'].value_counts().index)



plt.title("Top 15 heights", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)
top_15 = list(train_csv['longitude'].value_counts().head(15).reset_index()['index'])

data = train_csv[train_csv['longitude'].isin(top_15)]



# === PLOT ===

bird = mpimg.imread('../input/birdcall-recognition-data/orangebird.jpeg')

imagebox = OffsetImage(bird, zoom=0.6)

xy = (0.5, 0.7)

ab = AnnotationBbox(imagebox, xy, frameon=False, pad=1, xybox=(12.2, 7000))



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['longitude'], palette='cool', order = data['longitude'].value_counts().index)

ax.add_artist(ab)



plt.title("Top 15 longitude", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("longitude")
train_csv['duration_interval'] = ">500"

train_csv.loc[train_csv['duration'] <= 100, 'duration_interval'] = "<=100"

train_csv.loc[(train_csv['duration'] > 100) & (train_csv['duration'] <= 200), 'duration_interval'] = "100-200"

train_csv.loc[(train_csv['duration'] > 200) & (train_csv['duration'] <= 300), 'duration_interval'] = "200-300"

train_csv.loc[(train_csv['duration'] > 300) & (train_csv['duration'] <= 400), 'duration_interval'] = "300-400"

train_csv.loc[(train_csv['duration'] > 400) & (train_csv['duration'] <= 500), 'duration_interval'] = "400-500"



bird = mpimg.imread('../input/birdcall-recognition-data/yellow birds.jpg')

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
base_dir = '../input/birdsong-recognition/train_audio/'

train_csv['full_path'] = base_dir + train_csv['ebird_code'] + '/' + train_csv['filename']



# Now let's sample a fiew audio files

amered = train_csv[train_csv['ebird_code'] == "amered"].sample(1, random_state = 33)['full_path'].values[0]

cangoo = train_csv[train_csv['ebird_code'] == "cangoo"].sample(1, random_state = 33)['full_path'].values[0]

haiwoo = train_csv[train_csv['ebird_code'] == "haiwoo"].sample(1, random_state = 33)['full_path'].values[0]

pingro = train_csv[train_csv['ebird_code'] == "pingro"].sample(1, random_state = 33)['full_path'].values[0]

vesspa = train_csv[train_csv['ebird_code'] == "vesspa"].sample(1, random_state = 33)['full_path'].values[0]



bird_sample_list = ["amered", "cangoo", "haiwoo", "pingro", "vesspa"]
ipd.Audio(amered)
ipd.Audio(cangoo)
ipd.Audio(haiwoo)
ipd.Audio(pingro)
ipd.Audio(vesspa)
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

librosa.display.waveplot(y = audio_amered, sr = sr_amered, color = "cyan", ax=ax[0])

librosa.display.waveplot(y = audio_cangoo, sr = sr_cangoo, color = "orange", ax=ax[1])

librosa.display.waveplot(y = audio_haiwoo, sr = sr_haiwoo, color = "yellow", ax=ax[2])

librosa.display.waveplot(y = audio_pingro, sr = sr_pingro, color = "green", ax=ax[3])

librosa.display.waveplot(y = audio_vesspa, sr = sr_vesspa, color = "purple", ax=ax[4]);

for i, name in zip(range(5), bird_sample_list):

    ax[i].set_ylabel(name, fontsize=13)