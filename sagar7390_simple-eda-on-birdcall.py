import os



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import matplotlib.image as mpimg

from matplotlib.offsetbox import AnnotationBbox, OffsetImage





import plotly.express as px





import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon





import librosa

import librosa.display

import IPython.display as ipd



import sklearn



import warnings

warnings.filterwarnings('ignore')





import datetime as dt

from datetime import datetime   



import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import iplot



# Settings for pretty nice plots

plt.style.use('fivethirtyeight')

plt.show()



import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

cf.set_config_file(world_readable=True, theme='pearl')
# Import data

train_csv = pd.read_csv("../input/birdsong-recognition/train.csv")

test_csv = pd.read_csv("../input/birdsong-recognition/test.csv")



train_csv.head()

test_csv.head()



print("There are {} rows and {} columns in train file".format(train_csv.shape[0],train_csv.shape[1]))

print("There are {} rows and {} columns in test file".format(test_csv.shape[0],test_csv.shape[1]))

print(train_csv.shape[0])

print(train_csv.info())
print("There are {:,} unique bird species in the dataset.".format(len(train_csv['species'].unique())))
train_csv['year'] = train_csv['date'].apply(lambda x: x.split('-')[0])

plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['year'], palette="RdYlGn")



plt.title("Year of recording", fontsize=16)

plt.xticks(rotation=90, fontsize=13)

plt.yticks(fontsize=13)

plt.ylabel("Count", fontsize=14)

plt.xlabel("");
# Top 20 most common elevations

top_20 = list(train_csv['elevation'].value_counts().head(20).reset_index()['index'])

data = train_csv[train_csv['elevation'].isin(top_20)]



plt.figure(figsize=(16, 6))

ax = sns.countplot(data['elevation'], palette="RdYlGn", order = data['elevation'].value_counts().index)





plt.title("Top 20 Elevation Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Create data

data = train_csv['bird_seen'].value_counts().reset_index()





plt.figure(figsize=(16, 6))

ax = sns.barplot(x = 'bird_seen', y = 'index', data = data, palette="RdYlGn")





plt.title("Song was Heard, but was Bird Seen?", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Top 20 most common elevations

top_20 = list(train_csv['country'].value_counts().head(20).reset_index()['index'])

data = train_csv[train_csv['country'].isin(top_20)]





plt.figure(figsize=(16, 6))

ax = sns.countplot(data['country'], palette='RdYlGn', order = data['country'].value_counts().index)





plt.title("Top 20 Countries with most Recordings", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
# Creating Interval for *duration* variable

train_csv['duration_interval'] = ">500"

train_csv.loc[train_csv['duration'] <= 100, 'duration_interval'] = "<=100"

train_csv.loc[(train_csv['duration'] > 100) & (train_csv['duration'] <= 200), 'duration_interval'] = "100-200"

train_csv.loc[(train_csv['duration'] > 200) & (train_csv['duration'] <= 300), 'duration_interval'] = "200-300"

train_csv.loc[(train_csv['duration'] > 300) & (train_csv['duration'] <= 400), 'duration_interval'] = "300-400"

train_csv.loc[(train_csv['duration'] > 400) & (train_csv['duration'] <= 500), 'duration_interval'] = "400-500"





plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['duration_interval'], palette="RdYlGn")



plt.title("Distribution of Recordings Duration", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
print(train_csv['species'].value_counts())



train_csv['species'].value_counts().iplot()
from plotly.offline import iplot

# Total number of people who provided the recordings

print(train_csv['recordist'].nunique())

# Top 10 recordists in terms of the number of recordings done

train_csv['recordist'].value_counts()[:10].sort_values().iplot(kind='barh',color='#3780BF')
plt.figure(figsize=(16, 6))

ax = sns.countplot(train_csv['file_type'], palette = "RdYlGn", order = train_csv['file_type'].value_counts().index)





plt.title("Recording File Types", fontsize=16)

plt.ylabel("Frequency", fontsize=14)

plt.yticks(fontsize=13)

plt.xticks(rotation=45, fontsize=13)

plt.xlabel("");
train_csv['file_type'].value_counts()
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
TRAIN_EXT_PATH = "../input/xeno-canto-bird-recordings-extended-a-m/train_extended.csv"

train_ext = pd.read_csv(TRAIN_EXT_PATH)

train_ext.head()
len(train_ext['ebird_code'].value_counts())
len(train_ext)
df_original = train_csv.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "original_recordings"})

df_extended = train_ext.groupby("species")["filename"].count().reset_index().rename(columns = {"filename": "extended_recordings"})



df = df_original.merge(df_extended, on = "species", how = "left").fillna(0)

df["total_recordings"] = df.original_recordings + df.extended_recordings

df = df.sort_values("total_recordings").reset_index().sort_values('total_recordings',ascending=False)

df.head()
# Plot the total recordings

f, ax = plt.subplots(figsize=(10, 50))



sns.set_color_codes("pastel")

sns.barplot(x="total_recordings", y="species", data=df,

            label="total_recordings", color="r")



# Plot the original recordings

sns.set_color_codes("muted")

sns.barplot(x="original_recordings", y="species", data=df,

            label="original_recordings", color="g")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)

ax.set(xlim=(0, 2000), ylabel="",

       xlabel="Count")

sns.despine(left=True, bottom=True)