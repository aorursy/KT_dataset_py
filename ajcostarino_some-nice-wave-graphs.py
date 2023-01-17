import os



# Data processing

import pandas as pd

import numpy as np

import librosa

import IPython.display as ipd

import plotly.graph_objects as go



## Paths

DIR = '../'



INPUT   = DIR + 'input/'

LIB     = DIR + 'lib/'

WORKING = DIR + 'working/'



# Internal

BIRDSONG_RECOG = INPUT + 'birdsong-recognition' + '/'



BIRDSONG_TRAIN_AUDIO = BIRDSONG_RECOG + 'train_audio/'

EXAMPLE_TEST_AUDIO   = BIRDSONG_RECOG + 'example_test_audio/'



## Data structures



# Internal

BIRDS_LS = os.listdir(BIRDSONG_TRAIN_AUDIO)

TRAIN    = pd.read_csv(BIRDSONG_RECOG + 'train.csv')

TRAIN['path'] = BIRDSONG_RECOG + TRAIN['ebird_code'] + '/' + TRAIN['filename']



EXAMPLE_TEXT_AUDIO_METADATA = pd.read_csv(BIRDSONG_RECOG + 'example_test_audio_metadata.csv')

EXAMPLE_TEXT_AUDIO_SUMMARY = pd.read_csv(BIRDSONG_RECOG + 'example_test_audio_summary.csv')



BIRD_COLORS = {

 'squirrel' : 'burlywood',

 'brncre' : 'brown',

 'stejay' : 'steelblue',

 'mouqua' : 'firebrick',

 'rebsap' : 'darkred', 

 'unk'    : 'black',

 'hawo'   : 'darkgray',

 'daejun' : 'darkgoldenrod',

 'westan' : 'orange',

 'mouchi' : 'lightgray',

 'gockin' : 'forestgreen',

 'rebnut' : 'darkorange',

 'whhwoo' : 'red',

 'yerwar' : 'khaki',

}
def load_audio(path, sr=44100):

    x , sr = librosa.load(path, sr=sr)

    audio_data = (x, sr)

    return audio_data



def display_audio(path):

    return ipd.Audio(path)



def create_frame(audio_data):

    df = pd.DataFrame()

    df['signal']  = audio_data[0]

    df['seconds'] = df.index/audio_data[1]

    

    return df
def build_overall_scatter(signal, fig):

    line = {

        'color' : 'darkgray',

        'width' : 1

    }

    scatter = go.Scatter(

        x = signal['seconds'],

        y = signal['signal'],

        name = 'audio',

        line = line

        

    )

    

    fig = fig.add_trace(scatter)

    return fig





def build_birdcall_scatter(signal, s, e, fig):

    fil    = (signal['seconds'] >= s) & (signal['seconds'] <= e)

    signal = signal[fil]

    

    line = {

        'color' : 'rgb(84,84,84)',

        'width' : 1

    }

    

    scatter = go.Scatter(

        x = signal['seconds'],

        y = signal['signal'],

        line = line

        

    )

    

    fig = fig.add_trace(scatter)

    return fig





def build_birdcall_label(signal, birdcall, index, fig):

    mul = 1 if (index % 2 == 0) else -1

    s = birdcall['time_start']

    e = birdcall['time_end']

    color = BIRD_COLORS[birdcall['ebird_code']]

    scatter = go.Scatter(

        x = [s, e],

        y = [mul * 0.35, mul * 0.35],

    mode="lines",

    line=dict(color=color, width=2),

    )

    

    fig = fig.add_annotation(

        x=((e-s) / 2) + s,

        y=mul * 0.35 + mul * 0.075,

        xref="x",

        yref="y",

        text=birdcall['ebird_code'],

        showarrow=False,

        font=dict(

            family="Avenir",

            size=10,

            color="rgb(84,84,84)"

            ),

        align="center",

        arrowcolor="#636363",

        ax=30,

        ay=-30,

        bordercolor=BIRD_COLORS[birdcall['ebird_code']],

        borderwidth=2,

        borderpad=2,

        bgcolor="lightgray",

        opacity=1

        )

    

    fig = fig.add_trace(scatter)

    return fig





def update_x_axes(fig, s, e):

    fig = fig.update_xaxes(showgrid=False, range=[s, e + 2])

    return fig

    

    

def update_y_axes(fig, signal):

    upper = signal['signal'].max() + .25

    lower = signal['signal'].min() - .25

    fig = fig.update_yaxes(

        showgrid=True, 

        showticklabels=False,

        showline=True, 

        linewidth=2, 

        linecolor='rgb(80,80,80)', 

        gridwidth=2,  

        range=[lower, upper], 

        dtick=.2, 

        gridcolor='lightgray', 

        zeroline=True, 

        zerolinecolor='darkgray')

    return fig
def plot_interval(device, df, metadata, s, e):

    '''Just to make the graphing easier, only allow graphing 20 seconds at a time.

    '''

    if (e-s) > 20:

        raise Exception

    df_fil       = df[(df['seconds'] >= s) & (df['seconds'] <= e)]

    metadata_fil = metadata[(metadata['device'] == device) & 

                            (metadata['time_start'] <= e) & 

                            (metadata['time_end'] >= s)].reset_index(drop=True)

    

    # Basic titles and characteristics

    title       = f'{device} Interval: {s} - {e}s'

    xaxis_title = 'Amplitude'

    yaxis_title = 'Seconds'

    font        = {

        'family' : 'Avenir',

        'size'   : 12,

        'color'  : "#7f7f7f"

    }

    bgcolor = '#fff'

    

    # Create initial figure

    fig = go.Figure(

        layout=dict(

        title = title,

        xaxis_title = xaxis_title,

        yaxis_title = yaxis_title,

        font = font,

        plot_bgcolor = bgcolor,

        showlegend=False

        )

    )

    

    fig = build_overall_scatter(df_fil, fig)

    fig = update_x_axes(fig, s, e)

    fig = update_y_axes(fig, df_fil)

    

    for index, row in metadata_fil.iterrows():

        fig = build_birdcall_scatter(df_fil, row['time_start'], row['time_end'], fig)

        fig = build_birdcall_label(df_fil, row, index, fig)

    

    return fig
orange = load_audio(EXAMPLE_TEST_AUDIO + 'ORANGE-7-CAP_20190606_093000.pt623.mp3', sr=22050)

orange = create_frame(orange)

plot_interval('ORANGE-7-CAP', orange, EXAMPLE_TEXT_AUDIO_METADATA, 15, 35)
orange = load_audio(EXAMPLE_TEST_AUDIO + 'ORANGE-7-CAP_20190606_093000.pt623.mp3', sr=22050)

orange = create_frame(orange)

plot_interval('ORANGE-7-CAP', orange, EXAMPLE_TEXT_AUDIO_METADATA, 70, 90)
blkfr = load_audio(EXAMPLE_TEST_AUDIO + 'BLKFR-10-CPL_20190611_093000.pt540.mp3', sr=22050)

blkfr = create_frame(blkfr)

plot_interval('BLKFR-10-CPL', blkfr, EXAMPLE_TEXT_AUDIO_METADATA, 30, 50)