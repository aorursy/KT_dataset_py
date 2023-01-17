import librosa

import numpy as np

import pandas as pd

import seaborn as sns

import librosa.display

import plotly.express as px

import IPython.display as ipd



import matplotlib.pyplot as plt

%matplotlib inline

from matplotlib.offsetbox import AnnotationBbox

from mpl_toolkits.basemap import Basemap
train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

test = pd.read_csv('/kaggle/input/birdsong-recognition/test.csv')

audio_metadata = pd.read_csv('/kaggle/input/birdsong-recognition/example_test_audio_metadata.csv')

audio_summary = pd.read_csv('/kaggle/input/birdsong-recognition/example_test_audio_summary.csv')
train.shape
train.head().T
train['ebird_code'].nunique()
train['recordist'].nunique()
train['country'].nunique()
longitude = train[train["longitude"] != 'Not specified']['longitude'].apply(lambda x: float(x)).tolist()

latitude = train[train["latitude"] != 'Not specified']['latitude'].apply(lambda x: float(x)).tolist()
plt.figure(1, figsize=(16,6))

world_map = Basemap(projection='merc', llcrnrlat=-60, urcrnrlat=65, llcrnrlon=-180,

                    urcrnrlon=180, lat_ts=0, resolution='c')

world_map.fillcontinents(color='#191919',lake_color='#000000')

world_map.drawmapboundary(fill_color='#000000')

world_map.drawcountries(linewidth=0.1, color="w")

mxy = world_map(longitude, latitude)

world_map.scatter(mxy[0], mxy[1], s=3, c="#1292db", lw=0, alpha=1, zorder=5)

plt.title("Recording Locations")

plt.show()
country = train["country"].value_counts()

country = country[:35,]

plt.figure(figsize=(20, 6))

ax = sns.barplot(country.index, country.values, palette="hls")

plt.title("Country and Number of Audio File Recorded", fontsize=16)

plt.xticks(rotation=90, fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Number of Audio Files", fontsize=14)

plt.xlabel("");

for p in ax.patches:

    height = p.get_height()

    y=p.get_bbox().get_points()[1,1]

    ax.text(p.get_x()+p.get_width()/2., height + 350, int(y), ha="center", rotation=90)
plt.figure(figsize=(16, 6))

ax = sns.countplot(train['date'].apply(lambda x: x.split('-')[0]), palette="hls")

plt.title("Year of Recording", fontsize=16)

plt.xticks(rotation=90, fontsize=13)

plt.ylabel("Number of Recording", fontsize=14)

plt.xlabel("Years", fontsize=14)

for p in ax.patches:

    height = p.get_height()

    y=p.get_bbox().get_points()[1,1]

    ax.text(p.get_x()+p.get_width()/2., height + 50, int(y), ha="center", rotation=90)
train["file_type"].value_counts()
sound = train['type'].apply(lambda x: x.split(',')).reset_index().explode("type")

sound = sound['type'].apply(lambda x: x.strip().lower()).reset_index()

sound['type'] = sound['type'].replace('calls', 'call')

sound = sound['type'].value_counts()[:10,]

plt.figure(figsize=(16, 6))

ax = sns.barplot(sound.index, sound.values, palette="hls")

plt.title("Types of Sounds", fontsize=16)

plt.xticks(rotation=90, fontsize=13)

plt.yticks(fontsize=13)

plt.xlabel("");

for p in ax.patches:

    height = p.get_height()

    y=p.get_bbox().get_points()[1,1]

    ax.text(p.get_x()+p.get_width()/2., height + 350, int(y), ha="center", rotation=90)
data = train['bird_seen'].value_counts()

plt.figure(figsize=(16, 6))

ax = sns.barplot(data.index, data.values, palette="hls")

plt.title("Song was heard, but was Bird Seen?", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");

for p in ax.patches:

    height = p.get_height()

    y=p.get_bbox().get_points()[1,1]

    ax.text(p.get_x()+p.get_width()/2., height/2, int(y), ha="center", rotation=90)