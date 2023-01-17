import os

import glob



import librosa

import librosa.display



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from bokeh.models.tools import HoverTool

from bokeh.plotting import figure, output_notebook, show

from bokeh.models import ColumnDataSource

output_notebook()



import warnings

warnings.simplefilter('ignore')



pd.set_option('display.max_columns', None)
os.listdir("../input/birdsong-recognition")
df = pd.read_csv("../input/birdsong-recognition/train.csv", delimiter = ',')

print('There are Total {} datapoints in the dataset with {} Features'.format(df.shape[0], df.shape[1]))

df.head(3)
from IPython.display import IFrame

IFrame('https://ebird.org/species/{}'.format("aldfly"), width=800, height=450)
from IPython.display import IFrame

IFrame('https://ebird.org/species/{}'.format("baisan"), width=800, height=450)
from IPython.display import IFrame

IFrame('https://ebird.org/species/{}'.format("calqua"), width=800, height=450)
features_with_null = [feature for feature in df.columns if df[feature].isnull().sum()>0]

if features_with_null:

    print('Features with Null Values {}'.format(features_with_null))

else:

    print('Dataset contains no Null Values')
import missingno as msno

msno.bar(df)
print('Total Unique Bird Species : {} with Max No of any Bird Species as {} and Min No as {}'.format(len(df.ebird_code.unique()), df.ebird_code.value_counts().max(), df.ebird_code.value_counts().min()))
temp=df.ebird_code.value_counts().reset_index().rename(columns={"index": "ebird_code", "ebird_code": "Recording"})

temp["Species"]=df.species

temp=temp.sort_values("Recording")



Source=ColumnDataSource(temp)



tooltips = [

    ("Bird Code", "@Species"),

    ("Recordings Count", "@Recording")

]



fig1 = figure(plot_width = 800, plot_height = 4000,tooltips=tooltips, y_range = temp.ebird_code.values, title = "Count of Birds")

fig1.hbar("ebird_code", right = "Recording", source = Source, height = 0.4, color = "#03c2fc", alpha = 0.4)



fig1.xaxis.axis_label = "Recording Count"

fig1.yaxis.axis_label = "ebird_code"



show(fig1)
print('We have Total {} unique Users who provided the recordings'.format(len(df.recordist.unique())))
temp=df.recordist.value_counts()[:30].reset_index().rename(columns={"index": "Name", "recordist": "Recordings"})



Source=ColumnDataSource(temp)



tooltips = [

    ("Name of Recordist", "@Name"),

    ("No of Recordings", "@Recordings")

]



fig1 = figure(plot_width = 1000, plot_height = 400,tooltips=tooltips, x_range = temp["Name"].values, title = "Top-30 Recordists")

fig1.vbar("Name", top = "Recordings", source = Source, width = 0.4, color = "#03c2fc", alpha = 0.4)



fig1.xaxis.major_label_orientation = np.pi / 8

fig1.xaxis.axis_label = "Name of Recordist"

fig1.yaxis.axis_label = "Recordings"



show(fig1)
print('We have Total {} unique Location'.format(len(df.location.unique())))
import folium

from folium import plugins



df['latitude'] = df['latitude'].apply(lambda x : float(x) if '.' in x else None)

df['longitude'] = df['longitude'].apply(lambda x : float(x) if '.' in x else None)



try : 

    df.drop(['license', 'file_type'], inplace=True)

except :

    pass



m = folium.Map()



train_for_map = df[['latitude', 'longitude', 'species']].dropna()



# Marker Cluster

plugins.MarkerCluster(train_for_map[['latitude', 'longitude']].values,

                      list(train_for_map['species'].apply(str).values)

).add_to(m)



# Mouse Check

formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"

plugins.MousePosition(

    position='topright',

    separator=' | ',

    empty_string='NaN',

    lng_first=True,

    num_digits=20,

    prefix='Coordinates:',

    lat_formatter=formatter,

    lng_formatter=formatter,

).add_to(m)



# minimap

minimap = plugins.MiniMap()

m.add_child(minimap)





m
print('We have Total {} unique Countries'.format(len(df.country.unique())))
temp=df.groupby(["country","species"])["ebird_code"].count().reset_index()

for each_country in temp["country"].unique():

    a=temp[temp["country"]==each_country]

    

    Source=ColumnDataSource(a)

    

    tooltips = [

    ("Name of Bird Species", "@species"),

    ("Frequency", "@ebird_code")

    ]



    fig1 = figure(plot_width = 1000, plot_height = 400,tooltips=tooltips, x_range = a.species.values, title = "Bird distribution in {}".format(each_country))

    fig1.vbar("species", top = "ebird_code", source = Source, width = 0.4, color = "#03c2fc", alpha = 0.4)



    fig1.xaxis.major_label_orientation = np.pi / 8

    fig1.xaxis.axis_label = "Count"

    fig1.yaxis.axis_label = "Bird Species"



    show(fig1)



    
date=df.groupby('date')['ebird_code'].count().reset_index()

date
date.date = pd.to_datetime(date.date, errors = "coerce")

date.dropna(inplace = True)
Source = ColumnDataSource(date)



tooltips = [

    ("Date", "@date{%F}"),

    ("Recordings", "@ebird_code")

]



fig1 = figure(plot_width = 700, plot_height = 400, x_axis_type = "datetime", title = "Date of recording")

fig1.line("date", "ebird_code", source = Source, color = "#03c2fc", alpha = 0.4)



fig1.add_tools(HoverTool(tooltips=tooltips,formatters={"@date": "datetime"}))

fig1.xaxis.axis_label = "Year"

fig1.yaxis.axis_label = "Recordings"



show(fig1)
path = "../input/birdsong-recognition/train_audio/cacwre/XC11493.mp3"

import IPython.display as ipd

ipd.Audio(path)
def load_wave(files):   

    counter=1

    for each_file in files:      

        x , sr = librosa.load(each_file)

        plt.figure(figsize=(12, 6))

        plt.subplot(5, 1, counter)

        librosa.display.waveplot(x, sr=sr)

        counter+=1

        plt.title("FileName: {}, ebird_code : {}".format(each_file.split("/")[4],each_file.split("/")[5]))

        plt.plot()

        plt.tight_layout()
AUDIO_DIR="../input/birdsong-recognition/train_audio"

ebird_code="aldfly"



audio_files_path=glob.glob(AUDIO_DIR  +'/'+ ebird_code + '/*.mp3')[:5]

load_wave(audio_files_path)
def load_spec(files):   

    counter=1

    for each_file in files:      

        x , sr = librosa.load(each_file)

        plt.figure(figsize=(12, 12))

        plt.subplot(5, 1, counter)

        x = librosa.stft(x)

        x = librosa.amplitude_to_db(abs(x))

        librosa.display.specshow(x, sr=sr, x_axis='time', y_axis='hz')

        counter+=1

        plt.title("FileName: {}, ebird_code : {}".format(each_file.split("/")[4],each_file.split("/")[5]))

        plt.colorbar()

        plt.plot()

        plt.tight_layout()
AUDIO_DIR="../input/birdsong-recognition/train_audio"

ebird_code="aldfly"



audio_files_path=glob.glob(AUDIO_DIR  +'/'+ ebird_code + '/*.mp3')[:5]

load_spec(audio_files_path)
test_df=pd.read_csv("../input/birdsong-recognition/test.csv", delimiter=',')
test_df = pd.read_csv("../input/birdsong-recognition/test.csv", delimiter = ',')

print('There are Total {} datapoints in the dataset with {} Features:'.format(test_df.shape[0], test_df.shape[1]))

test_df
test_df_metadata = pd.read_csv('../input/birdsong-recognition/example_test_audio_metadata.csv')

print('There are Total {} datapoints in the dataset with {} Features'.format(test_df_metadata.shape[0], test_df_metadata.shape[1]))

test_df_metadata.head(3)
test_df_summary = pd.read_csv('../input/birdsong-recognition/example_test_audio_summary.csv')

print('There are Total {} datapoints in the dataset with {} Features'.format(test_df_summary.shape[0], test_df_summary.shape[1]))

test_df_summary.head(3)