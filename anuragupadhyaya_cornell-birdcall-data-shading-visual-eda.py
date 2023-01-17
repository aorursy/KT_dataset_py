# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

pd.options.display.max_columns = 50

import warnings

warnings.filterwarnings('ignore')

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline

import plotly.express as px



from collections import Counter



import plotly.graph_objects as go

from plotly.subplots import make_subplots



import holoviews as hv

from holoviews import opts



import datashader as ds, datashader.transfer_functions as tf, numpy as np

from datashader import spatial

import holoviews.operation.datashader as hd

from holoviews.operation import decimate



from functools import partial

import datashader as ds

from datashader.utils import export_image

from seaborn import color_palette

from holoviews.element.tiles import StamenTerrain, EsriTerrain



hv.extension('bokeh')
input_path = "/kaggle/input/birdsong-recognition/"



hv_opts = dict(cmap='jet', 

               bgcolor='aqua',

                fontsize={'xticks':7.7, 'yticks':7},

                xrotation=90,

                xaxis='top',

                yaxis='left',

                height=2200,

                width=1300,

                colorbar=True,

                tools=['hover'])



hv_bar = dict(fontsize={'xticks':7.7, 'yticks':7},

#               xrotation=90,

                xaxis='top',

                yaxis='left',

                height=2100,

                width=800,

                show_grid=True,

                invert_axes=True,

                tools=['hover'])



hv_subplot = dict(fontsize={'xticks':7.7, 'yticks':7},

#               xrotation=90,

#                 xaxis='top',

                yaxis='left',

                height=300,

                width=1100,

                show_grid=True,

                  shared_axes=False,

#                 invert_axes=True,

                tools=['hover'])



hv_spectra = dict(height=250,

                  width=550,

                  show_grid=True,

                  xaxis=None,

                  yaxis=None,

                  tools=['hover'])
#check audio files per bird-type

audio_path = os.path.join(input_path, "train_audio/")

audio_dist = {}

for bird_type in os.listdir(audio_path):

    len_audio = len(os.listdir(audio_path + os.sep + f"{bird_type}"))

    audio_dist[bird_type] = len_audio



audio_df = pd.DataFrame.from_dict(audio_dist, orient='index', \

                                  columns=['Audio_Count']).reset_index(drop=False).rename(columns={'index':'Bird_Type'})





audio_df.info()
hv.Bars(audio_df.sort_values(by='Audio_Count', ascending=False)).opts(**hv_bar, color='lightpink',

                                                                      title='Audio File Distribution For Different Birds.')
train_df = pd.read_csv(os.path.join(input_path, 'train.csv'))

train_df.head()
train_df.info()
lat_long_df = train_df[['longitude', 'latitude', 'country', 'species', 'ebird_code', 'duration', 'elevation']]

lat_long_df.replace('Not specified', np.NaN, inplace=True)

lat_long_df.replace('?', np.NaN, inplace=True)

lat_long_df.dropna(axis=0, inplace=True)

lat_long_df['longitude'] = lat_long_df['longitude'].apply(lambda x: float(x))

lat_long_df['latitude'] = lat_long_df['latitude'].apply(lambda x: float(x))

lat_long_df[['country', 'species']] = lat_long_df[['country', 'species']].apply(lambda x: x.astype('category'))



#generate Web Mercator format for Latitude and Longitude..

from datashader.utils import lnglat_to_meters as webm

lat_long_webm = list(lat_long_df[['longitude', 'latitude']].apply(lambda x: webm(*x), axis=1).values)

lat_long_df.loc[:, 'long_wemr'] = [i[0] for i in lat_long_webm]

lat_long_df.loc[:, 'lat_wemr'] = [i[1] for i in lat_long_webm]
decimate.max_samples=10

x_range,y_range = (-19230442.03453801,  19831389.17363642), (-6933173.79129572, 15142823.60169782)



plot_width  = int(1300)

plot_height = int(800)



unique_values = lat_long_df['ebird_code'].unique()

colors = ['#%02x%02x%02x' % (a, b, c) for a,b,c in np.round(255*np.array(color_palette('plasma',n_colors=len(unique_values)))).astype(int)]

color_key = {val:color for val,color in zip(unique_values,colors)}
tiles = StamenTerrain().redim.range(x=tuple(x_range), y=tuple(y_range))

lat_longs = hv.Points(lat_long_df, ['long_wemr', 'lat_wemr']).opts(size=5, alpha=0.7)



shade = hd.datashade(lat_longs,

                     aggregator=ds.count_cat('ebird_code'),

                     color_key=color_key)



tiles * hd.dynspread(shade).opts(width=plot_width,

                                  height=plot_height,

                                  xaxis=None, yaxis=None)
def create_image(df, country_name, title=None, w=plot_width, h=plot_height, annotate=True):

    

    country_lat_long = lat_long_df[lat_long_df['country'] == country_name][['long_wemr', 'lat_wemr', 'ebird_code']]

    country_lat_long.reset_index(drop=True,inplace=True)

    country_species = country_lat_long.pop('ebird_code')

    

    (long_min, lat_min), (long_max, lat_max) = country_lat_long.min(), country_lat_long.max()

    

    longitude_range, latitude_range = (long_min, long_max), (lat_min, lat_max)

    x_range, y_range = longitude_range, latitude_range

    

    country_lat_long.loc[:, 'ebird_code'] = country_species.values



    tiles = EsriTerrain().redim.range(x=tuple(x_range), y=tuple(y_range))

    

    lat_longs = hv.Points(country_lat_long, ['long_wemr', 'lat_wemr']).opts(size=25, alpha=0.9)

    shade = hd.datashade(lat_longs,

                         aggregator=ds.count_cat('ebird_code'),

                         color_key=color_key)

    if annotate:

        labels = hv.Labels(country_lat_long, ['long_wemr', 'lat_wemr'], 'ebird_code').opts(opts.Labels(text_color='ebird_code',

                                                                                                        padding=5.5, 

                                                                                                        fontsize=1,

                                                                                                        text_alpha=0.4))

        layout = tiles * hd.dynspread(shade).opts(width=w,title=title,

                                                  fontsize=13,

                                                height=h,

                                                xaxis=None,

                                                yaxis=None) * decimate(labels)

        return layout

    

    else:

        layout = tiles * hd.dynspread(shade).opts(width=w,title=title,

                                                height=h,

                                                xaxis=None,

                                                yaxis=None)

        return layout
top_5 = lat_long_df['country'].value_counts()[:5].index.to_list()

country_layout = []



for country in top_5:

    country_layout.append(create_image(lat_long_df, str(country), title=str(country), w=700, h=500, annotate=True))

    

layout = hv.Layout(country_layout).cols(2)



display(layout)
cat_unique_df = train_df.select_dtypes(include='object').nunique().reset_index().rename(columns={'index':'Column_Name',

                                                                                 0 : 'Unique_values'}).sort_values(by='Unique_values')

hv.Bars(cat_unique_df).opts(**hv_bar, color='aqua', title='Unique Values For Each Catergorical Variable.')
int_unique_df = train_df.select_dtypes(include=['int', 'float']).nunique().reset_index().rename(columns={'index':'Column_Name',

                                                                                                   0 : 'Unique_values'}).sort_values(by='Unique_values')

hv.Bars(int_unique_df).opts(**hv_subplot,

                            color='lightgreen',

#                             height=500,

                            title='Unique Values For Each Integer/Float Variable.')
hv.Bars(train_df['species'].value_counts()).opts(**hv_bar, color='orange', title='Distribution of Species.')
hv.Bars(train_df['ebird_code'].value_counts()).opts(**hv_bar, color='orange', title='Distribution of ebird_code.')
hv.Bars(train_df['rating'].value_counts()).opts(**hv_subplot, color='lightblue', title='Distribution of Ratings.')
hv.Bars(train_df['sampling_rate'].value_counts()).opts(**hv_subplot, color='lightblue', title='Distribution of Sampling Rate for the Audio Files.')
hv.Bars(train_df['playback_used'].value_counts()).opts(**hv_subplot, color='lightblue', title='Distribution of Playback Audio.')
hv.Bars(train_df['number_of_notes'].value_counts()).opts(**hv_subplot, color='lightblue', title='Distribution Of Number Of Notes in Audio.')
hv.Bars(train_df['playback_used'].value_counts()).opts(**hv_subplot, color='lightblue', title='Distribution Of Playback Used.')
df_date = train_df.groupby("date")["species"].count().reset_index().rename(columns = {"species": "recordings"})

df_date.date = pd.to_datetime(df_date.date, errors = "coerce")

df_date["weekday"] = df_date.date.dt.day_name()

df_date.dropna(inplace = True)

per_day_records = df_date.groupby('weekday', as_index=False).sum().sort_values(by='weekday')
sub_1 = hv.Curve(data=df_date).opts(**hv_subplot, color='darkgrey', title='Yearwise Recordings')

sub_2 = hv.Bars(data=per_day_records).opts(**hv_subplot, color='grey', title='Daywise Recordings')

hv.Layout([sub_1, sub_2]).cols(1)
hv.BoxWhisker(train_df, vdims='duration', kdims='species').opts(**hv_bar, title='Distribution of Duration Of Audio \n wrt. Bird Species.')
countrywise_species_df = train_df.groupby(['country', 'species'], as_index=False)['ebird_code'].count()

hv_opts['cmap'] = 'viridis'

hv.HeatMap(countrywise_species_df).opts(**hv_opts, title='Countrywise Bird Species Distribution.')
import librosa

import random



def get_file(n=1, species=5):

    ran_samples = {}

    

    for species in list(audio_dist.keys())[:5]:

        species_samples = os.listdir(audio_path + os.sep + species)

        ran_samples[species] = random.sample(species_samples, n).pop()

    

    return [audio_path + sp + os.sep + file for sp, file in ran_samples.items()]

    



sample_files = get_file(n=1,species=5)

print(sample_files)
tempogram_info = {}

chromagram_info = {}

spectral_bandwidth_info = {}

tonnetz_info = {}

mfcc_info = {}

poly_info = {}

spec_contrast_info = {}

fourier_tempo_info = {}





for file in sample_files:

    print(file)

    data, sr = librosa.load(file)

    

    chromagram_info[file] = librosa.feature.chroma_stft(data, sr=sr)

    spectral_bandwidth_info[file] = librosa.feature.spectral_bandwidth(data, sr=sr)

    tonnetz_info[file] = librosa.feature.tonnetz(data, sr=sr)

    mfcc_info[file] = librosa.feature.mfcc(data, sr=sr)

    poly_info[file] = librosa.feature.poly_features(data, win_length=15, sr=sr)

    spec_contrast_info[file] = librosa.feature.spectral_contrast(data, sr=sr)

    

    #declare onset strength with hop length for rythmic features aka tempogram..

    oenv = librosa.onset.onset_strength(y=data, sr=sr, hop_length=512)

    fourier_tempo_info[file] = np.abs(librosa.feature.fourier_tempogram(onset_envelope=oenv,

                                                                        sr=sr,

                                                                        hop_length=512))

    tempogram_info[file] = librosa.feature.tempogram(onset_envelope=oenv,

                                                     sr=sr,

                                                     hop_length=512)
def plot_features(features_dict, title='Chromagram'):

    layout = []



    for k,v in features_dict.items():

        species, files = k.split("/")[-2:]

        gram = hv.Image(features_dict[k]).opts(**hv_spectra, cmap='plasma',

                                               title=f"{species.capitalize()}-{files.capitalize()} || {title}")



        layout.append(gram)

    

    plot = hv.Layout(layout).cols(2)



    return plot
plot_features(chromagram_info)
plot_features(tonnetz_info, title='Tonnetz - Tonal Centroid.')
plot_features(mfcc_info, title='MFCCs.')
plot_features(poly_info, title='Poly Feats. window size 15')
plot_features(spec_contrast_info, title='Spectral Contrast.')
plot_features(tempogram_info, title='Auto-Correlation Tempogram')
plot_features(fourier_tempo_info, title='Fourier Tempogram.')
#let's use one audio file



sample_audio = sample_files[0]

sample_audio, rate = librosa.load(sample_audio)



#spectrogram ..

sample_stft = np.abs(librosa.stft(sample_audio))

#decompose the spectrogram such that components.dot(activations)..

comps, acts = librosa.decompose.decompose(sample_stft, n_components=32)



#reconstructed...

stft_recons = comps.dot(acts)
stft_glyph = hv.Raster(librosa.amplitude_to_db(sample_stft,

                                               ref=np.max)).opts(**hv_subplot,

                                                                              cmap='plasma',

                                                                              title="Spectrogram")



#decompose..

comps_glyph = hv.Raster(librosa.amplitude_to_db(comps,

                                                ref=np.max)).opts(**hv_subplot,

                                                                         cmap='plasma',

                                                                         title='Components')

acts_glyph = hv.Image(acts).opts(**hv_subplot,

                                 cmap='plasma',

                                 title='Activations')



#reconstruct..

stft_recons_glyph = hv.Raster(librosa.amplitude_to_db(stft_recons,

                                                      ref=np.max)).opts(**hv_subplot,

                                                                                     cmap='plasma',

                                                                                     title='Reconstructed Spectogram | [coms.dot(actss)]')



hv.Layout(stft_glyph + comps_glyph + acts_glyph + stft_recons_glyph).cols(1) 