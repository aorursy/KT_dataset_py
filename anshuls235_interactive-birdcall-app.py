!pip install pydub
#Libraries Required

import pandas as pd

import numpy as np

import plotly.graph_objects as go

import os

import requests

import re

import IPython.display as ipd

from pydub import AudioSegment

import requests

import json

from bs4 import BeautifulSoup

import ipywidgets as widgets
"""

dic_bird = {} 

for code in df_train.ebird_code.unique():

    if code not in dic_bird:

        url = 'https://ebird.org/species/'+code

        r = requests.get(url)

        soup = BeautifulSoup(r.content, 'html.parser')

        desc = soup.find('meta',property="og:description")

        image = soup.find('meta',property="og:image")

        dic_bird[code]={}

        dic_bird[code]['description'] = desc['content']

        dic_bird[code]['image'] = image['content']

#Convert the dictionary to dataframe.

df = pd.DataFrame(columns=['species','description','image'])

i = 0

for key,val in dic_bird.items():

    df.loc[i,'species'] = key

    df.loc[i,'description'] = val['description']

    df.loc[i,'image'] = val['image']

    i+=1

df.to_csv('bird_details.csv',index=False)

"""
#Get the training data and the dataset containing images & descriptions

df_train = pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')

df = pd.read_csv('/kaggle/input/birdsongrecognitiondetails/bird_details.csv')

media_path = '/kaggle/input/birdsong-recognition/train_audio/'
# Dictionary to Store Default values

default_values = {

    'species': 'Alder Flycatcher',

    'recordist': 'Jonathon Jongsma',

    'audio': 'XC134874.mp3',

    'image': df.loc[df['species']=='aldfly','image'].tolist()[0],

    'text': df.loc[df['species']=='aldfly','description'].tolist()[0],

    'filename': os.path.join('/kaggle/input/birdsong-recognition/train_audio/','aldfly','XC134874.mp3'),

    'latitude': 44.793,

    'longitude': -92.962,

    'location': 'Grey Cloud Dunes SNA, Washington, Minnesota'

}



def get_recordist_options(species):

    """Getter method for Recordist values."""

    return df_train[df_train['species']==species]['author'].unique().tolist()



def get_audio_options(species,recordist):

    """Getter method for Audio file values."""

    return df_train[(df_train['species']==species)&(df_train['author']==recordist)]['filename'].unique().tolist()



species = widgets.Dropdown(

    description = 'Species:   ',

    value = default_values['species'],

    options = df_train['species'].unique().tolist(),

    layout=dict(width='233px')

)



recordists = widgets.Dropdown(

    description = 'Recordist:   ',

    value = default_values['recordist'],

    options = get_recordist_options(default_values['species']),

    layout=dict(width='233px')

)



audios = widgets.Dropdown(

    description = 'Filename:    ',

    value = default_values['audio'],

    options = get_audio_options(default_values['species'],default_values['recordist']),

    layout=dict(width='233px')

)

container1 = widgets.HBox(children=[species, recordists, audios])

container1
title = widgets.HTML('<h1>{}</h1>'.format(default_values['species']),layout=dict(width='350px'))

out = widgets.Output(layout=dict(width='350px',margin='10px 0px 0px 0px'))

out.append_display_data(ipd.Audio(default_values['filename']))

container2 = widgets.HBox(children=[title,out])

container2
im = widgets.HTML('<img src="{}"/>'.format(default_values['image']),

                 layout=dict(height='250px',width='300px'))

text = widgets.HTML('<h5>{}</h5>'.format(default_values['text']),

                 layout=dict(height='250px',width='400px',margin='10px 0px 0px 10px'))

container3 = widgets.HBox(children=[im,text])

container3
def get_elevation(val):

    """Derive the elevation value from the string. Also, I have 

    kept negative elevation values as below sea level is also a possibility."""

    l = re.findall('[~\?]?(-?\d+[\.,]?\d*)-?(\d*)',val)

    val1=0

    val2=0

    if l:

        if l[0][0]:

            val1=float(l[0][0].replace(',',''))

        if l[0][1]:

            val2=float(l[0][1].replace(',',''))

        if val1!=0 and val2!=0:

            return (val1+val2)/2

        return val1

    else:

        return float('nan')

df_train.elevation=df_train.elevation.apply(lambda x: get_elevation(x))



def get_stats(species):

    """Get the Average rating,Duration of chip &

        elevation of the bird species."""

    df_sp = df_train[df_train['species']==species]

    avg_rating = np.round(df_sp.rating.mean(),2)

    avg_duration = np.round(df_sp.duration.mean(),2)

    avg_elevation = np.round(df_sp.elevation.mean(),2)

    return avg_rating,avg_duration,avg_elevation



r,d,e = get_stats(default_values['species'])

rating = widgets.HTML('<h2>Avg. Rating</h2><h4>{}</h4>'.format(r),layout=dict(width='233px'))

duration = widgets.HTML('<h2>Avg. Duration</h2><h4>{} s</h4>'.format(d),layout=dict(width='233px'))

elevation = widgets.HTML('<h2>Avg. Elevation</h2><h4>{} m</h4>'.format(e),layout=dict(width='233px'))

container4 = widgets.HBox(children=[rating,duration,elevation])

container4
def read(f, normalized=False):

    """Converts MP3 to numpy array"""

    a = AudioSegment.from_mp3(f)

    y = np.array(a.get_array_of_samples())

    if a.channels == 2:

        y = y.reshape((-1, 2))

    if normalized:

        return a.frame_rate, np.float32(y) / 2**15

    else:

        return a.frame_rate, y



def plot_waveform(arr,filename):

    """Plots the waveform from the numpy array"""

    fig = go.FigureWidget()

    try:

        channels = arr.shape[1]

        for channel in range(channels):

            fig.add_trace(go.Scatter(name='channel '+str(channel+1),y=arr[:,channel]))

    except IndexError:

        fig.add_trace(go.Scatter(y=arr,showlegend=False))

    fig.update_layout(template='seaborn',plot_bgcolor='rgb(255,255,255)',paper_bgcolor='rgb(255,255,255)',

                 height = 200, width = 700,title=filename,legend=dict(x=0.3,y=1.3,orientation='h'),

                 xaxis=dict(mirror=True,linewidth=2,linecolor='black'),

                 yaxis=dict(mirror=True,linewidth=2,linecolor='black'),

                 margin=dict(l=0,r=0,t=0,b=5))

    return fig

    

rate, arr = read(default_values['filename'])

title_g = widgets.HTML('<h2>Waveform of the bird chirp</h2>')

g = plot_waveform(arr,default_values['filename'].split('/')[-1])
#get the mapbox token

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()

secret_value_1 = user_secrets.get_secret("mapboxtoken")



def plot_location(lat,lon,location):

    """Plots the recording location on the map."""

    trace = go.Scattermapbox(lon=[lon],lat=[lat],hovertext=location,

                                           marker=dict(symbol='campsite',size=20,color='blue'))

    data = [trace]

    layout = go.Layout(

        width=700,

        height=200,

        margin=dict(l=0,r=0,t=5,b=0),

        hovermode='closest',

        mapbox=dict(

            accesstoken=secret_value_1,

            bearing=0,

            style='satellite-streets',

            center=go.layout.mapbox.Center(

                lat=lat,

                lon=lon

            ),

            pitch=0,

            zoom=10

        )

    )

    figure = go.Figure(data=data, layout=layout)

    fig = go.FigureWidget(figure)

    return fig



title_map = widgets.HTML('<h2>Where the recording was made?</h2>')

loc_map = plot_location(default_values['latitude'],default_values['longitude'],default_values['location']) 
app = widgets.VBox(children=[container1,container2,container3,container4,title_g,g,title_map,loc_map])
def get_filename(ebird_code,file):

    """Getter method for filename."""

    return os.path.join('/kaggle/input/birdsong-recognition/train_audio/',ebird_code,file)



def get_ebird_code(species):

    """Getter method for ebird_code."""

    return df_train.loc[df_train['species']==species,'ebird_code'].tolist()[0]



def get_image(ebird_code):

    """Getter method for image URL."""

    return df.loc[df['species']==ebird_code,'image'].tolist()[0]



def get_text(ebird_code):

    """Getter method for species description."""

    return df.loc[df['species']==ebird_code,'description'].tolist()[0]



def response_sp(change):

    """callback function for species dropdown"""

    ecode = get_ebird_code(species.value)

    

    #change recordists dropdown

    options = get_recordist_options(species.value)

    recordists.options = options

    recordists.value = options[0]

    

    #change filename dropdown

    options = get_audio_options(species.value,recordists.value)

    audios.options = options

    audios.value = options[0]

    

    #change title, image and text

    title.value = '<h1>{}</h1>'.format(species.value)

    im.value = '<img src="{}"/>'.format(get_image(ecode))

    text.value = '<h5>{}</h5>'.format(get_text(ecode))

    

    #change bird stats

    r,d,e = get_stats(species.value)

    rating.value = '<h2>Avg. Rating</h2><h4>{}</h4>'.format(r)

    duration.value = '<h2>Avg. Duration</h2><h4>{} s</h4>'.format(d)

    elevation.value = '<h2>Avg. Elevation</h2><h4>{} m</h4>'.format(e)



def response_re(change):

    """callback function for recordists dropdown."""

    #change audios dropdown

    options = get_audio_options(species.value,recordists.value)

    audios.options = options

    audios.value = options[0]



def response_au(change):

    """callback function for audios dropdown."""

    ecode = get_ebird_code(species.value)

    file = audios.value

    filename = get_filename(ecode,file)

    with out:

        ipd.clear_output()

        ipd.display(ipd.Audio(filename))

    rate, arr = read(filename)

    g_new = plot_waveform(arr,filename.split('/')[-1])

    g.update(data=g_new.data,layout=g_new.layout)

    lat = df_train.loc[df_train['filename']==audios.value,'latitude'].tolist()[0]

    lon = df_train.loc[df_train['filename']==audios.value,'longitude'].tolist()[0]

    location = df_train.loc[df_train['filename']==audios.value,'location'].tolist()[0]

    loc_map_new = plot_location(float(lat),float(lon),location)

    loc_map.update(data=loc_map_new.data,layout=loc_map_new.layout)



#Definition of callbacks    

species.observe(response_sp, names="value")

recordists.observe(response_re, names="value")

audios.observe(response_au, names="value")
#Run the App!

app