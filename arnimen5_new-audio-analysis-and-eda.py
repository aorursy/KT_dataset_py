import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import librosa

from collections import Counter

import plotly.express as px

from plotly import graph_objs as go



import random



import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow.keras.applications import ResNet50



import sklearn

from pylab import *

from scipy import *
PATH = '../input/birdsong-recognition/'

train_df =  pd.read_csv(PATH + 'train.csv')

test_df = pd.read_csv(PATH + 'test.csv')

example_test_audio_metadata = pd.read_csv(PATH + 'example_test_audio_metadata.csv')

example_test_audio_summary = pd.read_csv(PATH + 'example_test_audio_summary.csv')
def info_df(df):

    if 'ebird_code' in df.columns:

        print('In train dataset: ')

    else:

        print('In test dataset: ')

    print('Count of columns {}'.format(df.shape[1]))

    print('String of columns {}'.format(df.shape[0]))

info_df(train_df)



info_df(test_df)
train_df.head(3)
top = Counter([ i for i in train_df['ebird_code']])

temp = pd.DataFrame(top.most_common(25))

temp.columns = ['Most_bird','count']

temp.style.background_gradient(cmap='Reds')

top = Counter([ i for i in train_df['ebird_code']])

temp = pd.DataFrame(top.most_common()[:-25:-1])

temp.columns = ['Least_bird','count']

temp.style.background_gradient(cmap='Blues')
top = Counter([ i for i in train_df['ebird_code']])

temp = pd.DataFrame(top.most_common(270))



temp.columns = ['Most_bird','count']

fig = px.bar(temp, x="count", y="Most_bird", title='Distribution of birds in our dataset', orientation='h', 

             width=900, height=900, color='Most_bird')

fig.show()
plt.figure(figsize=(10, 6))

ax = sns.barplot(x = 'channels', y = 'ebird_code', data = pd.DataFrame(train_df['ebird_code'].groupby(train_df['channels']).count()).reset_index(), palette="muted")

plt.title('Count of stereo and mono recordings', fontsize=16)

plt.xlabel("");
top = Counter([ i for i in train_df['country']])

temp = pd.DataFrame(top.most_common(25))

temp.columns = ['Recordings per Country','count']

temp.style.background_gradient(cmap='Blues')
temp = pd.DataFrame(top.most_common(100))

temp.columns = ['Most_common_countries','count']

fig = px.bar(temp, x="count", y="Most_common_countries", title='Recordings per Country', orientation='h', 

             width=900, height=900, color='Most_common_countries')

fig.show()


df = px.data.gapminder().query("year==2007")[["country", "iso_alpha"]]



data = pd.merge(left=train_df, right=df, how="inner", on="country")



# Group by country and count how many species can be found in each

data = data.groupby(by=["country", "iso_alpha"]).count()["species"].reset_index()



fig = px.choropleth(data, locations="iso_alpha", color="species", hover_name="country",

                    color_continuous_scale=px.colors.sequential.Purpor,

                    title = "World Map: Recordings per Country")

fig.show()
def bird_countries(view, df_view):

    '''

    input - ebird code, dataframe

    output - distribution of bird around the countries

    '''

    df_view = df_view.loc[df_view['ebird_code']==view] 

    df = px.data.gapminder().query("year == 2007")[["country", "iso_alpha"]]

    data = pd.merge(left=df_view, right=df, how="inner", on="country")

    data = data.groupby(by=["country", "iso_alpha"]).count()["species"].reset_index()

    fig = px.scatter_geo(data, locations="iso_alpha",

                     color="species", # which column to use to set the color of markers

                     hover_name="country", # column added to hover information

                     projection="natural earth",

                     title ="World Map: {} per Country".format(view))

    fig.show() 



    

    

    
bird_countries('houspa', train_df)

bird_countries('carwre', train_df)

bird_countries('amepip', train_df)
df_country = pd.DataFrame(train_df['ebird_code'].groupby(train_df['country']).unique()).reset_index()

top = Counter([item for sublist in df_country['ebird_code'] for item in sublist])

temp = pd.DataFrame(top.most_common(25))

temp.columns = ['bird_per_Countries','count']

temp.style.background_gradient(cmap='Greens')
df_country = pd.DataFrame(train_df['ebird_code'].groupby(train_df['country']).unique()).reset_index()

top = Counter([item for sublist in df_country['ebird_code'] for item in sublist])

temp = pd.DataFrame(top.most_common()[:-25:-1])

temp.columns = ['bird_per_Countries','count']

temp.style.background_gradient(cmap='Reds')
temp = temp.loc[temp['count'] == 1]

temp['ebird_code'] = temp['bird_per_Countries']

data = pd.merge(left=temp, right=train_df, how="inner", on="ebird_code")[['ebird_code','country']]

pd.DataFrame(data['country'].groupby(data['ebird_code']).unique())
def bird_location(view, df_view):

    '''

    input - ebird code, dataframe

    output - bird location

    '''

    df_view = df_view.loc[df_view['ebird_code'] == view][['ebird_code','latitude','longitude']]

    df_view = df_view.loc[df_view['longitude'] != 'Not specified']

    df_view = df_view.loc[df_view['latitude'] != 'Not specified']

    df_view['longitude'] = df_view.longitude.astype('float')

    df_view['latitude'] = df_view.latitude.astype('float')

    px.set_mapbox_access_token("pk.eyJ1IjoiYXJuaW1lbjUiLCJhIjoiY2tlM2U3a3EwMGliZzJ5bXNnYjE2YTJrciJ9.4GEtm-YYF0e0nIzyoSeABw")

    fig = px.scatter_mapbox(df_view, lat="latitude", lon="longitude",  color="latitude", 

                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=2,

                            title ='latitude and longitude of the {} record'.format(view))

    fig.show()

bird_location('houspa', train_df)

bird_location('carwre', train_df)

bird_location('amepip', train_df)

audio_path = []

for i in train_df.index:

    initial_letter = train_df.loc[i]['ebird_code'][0] 

    if initial_letter < 'c':

        audio_path.append('../input/birdsong-resampled-train-audio-00/')

    elif initial_letter < 'g':

        audio_path.append('../input/birdsong-resampled-train-audio-01/')

    elif initial_letter < 'n':

        audio_path.append('../input/birdsong-resampled-train-audio-02/')

    elif initial_letter < 's':

        audio_path.append('../input/birdsong-resampled-train-audio-03/')

    else:

        audio_path.append('../input/birdsong-resampled-train-audio-04/')

        

train_df['audio_path'] = audio_path
def random_audio(df):

    index = random.choice(list(df.index))

    name = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

    y, sr = librosa.load(name)

    print('y:', y, '\n')

    print('y shape:', np.shape(y), '\n')

    print('Sample Rate (KHz):', sr, '\n')

    print('Check Len of Audio:', np.shape(y)[0]/sr)

    return y, sr 



y, sr = random_audio(train_df)
from scipy.io import wavfile as wav

import scipy



def anlysis_signal(df):

    index = random.choice(list(df.index))

    name = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

    M=501

    fig = plt.figure(figsize=(25,12))

    hM1=int(np.floor((1+M)/2))

    hM2=int(np.floor(M/2))

    (fs,x)=wav.read(name)

    x1=x[5000:5000+M]*np.hamming(M)

    N=511

    fftbuffer=np.zeros([N])

    fftbuffer[:hM1]=x1[hM2:]

    fftbuffer[N-hM2:]=x1[:hM2]

    X=scipy.fft.fft(fftbuffer)

    mX=abs(X)

    pX=np.angle(X)

    suptitle("Signal analysis {}".format(df.loc[index]['filename'].split('.')[0]))

    subplot(3, 1, 1)

    st='input signal {}'.format(df.loc[index]['ebird_code'])

    plt.title(st, fontsize=16)

    plot(x,linewidth=2, c = 'green')

    legend(loc='center')

    subplot(3, 1, 2)

    st='Frequency spectrum of the input signal'

    plt.title(st, fontsize=16)

    plot(mX,linewidth=2, c = 'red')

    legend(loc='best')

    subplot(3, 1, 3)

    st='Phase spectrum of the input signal'

    pX=np.unwrap(np.angle(X))

    plt.title(st, fontsize=16)

    plot(pX,linewidth=2)

    legend(loc='best') 

    show()
for i in range(3):

    anlysis_signal(train_df)
def audios_ebird(label, df):

    df = df.loc[df['ebird_code'] == label]

    index = random.choice(df.index)

    name = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

    y, sr = librosa.load(name)

    return y, sr





color = ['red', 'green', 'yellow', 'orange', 'blue']

def visualization_audio_bird(label, df):

    col = random.choice(color)

    fig = plt.figure(figsize=(25,12))

    df = df.loc[df['ebird_code'] == label][['ebird_code','filename','audio_path']]

    fig.suptitle(label, fontsize=30, c=col)

    num = 0

    for index in df.index:

        num += 1

        if num > 20:

            break

        plt.subplot(5,4,num)

        filepath = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

        clip, sample_rate = librosa.load(filepath, sr=None)

        plt.axis('off')

        plt.plot(clip, c=col, lw=0.5)

        

    
visualization_audio_bird('aldfly', train_df)

visualization_audio_bird('osprey', train_df)

visualization_audio_bird('coohaw', train_df)
def random_audio_sample_rate(df):

    index = random.choice(list(df.index))

    name = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

    librosa_audio, librosa_sample_rate = librosa.load(name)

    scipy_sample_rate, scipy_audio = wav.read(name)

    print("Original sample rate: {}".format(scipy_sample_rate))

    print("Librosa sample rate: {}".format(librosa_sample_rate))
random_audio_sample_rate(train_df)
def audio_file_min_max(df):

    index = random.choice(list(df.index))

    name = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

    librosa_audio, librosa_sample_rate = librosa.load(name)

    scipy_sample_rate, scipy_audio = wav.read(name)

    print('Original audio file min~max range: {} to {}'.format(np.min(scipy_audio), np.max(scipy_audio)))

    print('Librosa audio file min~max range: {0:.2f} to {0:.2f}'.format(np.min(librosa_audio), np.max(librosa_audio)))
audio_file_min_max(train_df)
import librosa

import librosa.display



def mfccs(df):

    index = random.choice(list(df.index))

    name = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

    librosa_audio, librosa_sample_rate = librosa.load(name)

    mfccs = librosa.feature.mfcc(y=librosa_audio, sr=librosa_sample_rate, n_mels = 128, fmin=20, fmax=16000)

    plt.figure (figsize = (8,8))

    librosa.display.specshow(mfccs, sr = librosa_sample_rate, x_axis = 'time')

    plt.title('MFCC')

    return mfccs
mfcc = mfccs(train_df)

print(mfcc.shape)
def linear_spectrogram(df):

    index = random.choice(list(df.index))

    name = '{}{}/{}.wav'.format(df.loc[index]['audio_path'] , df.loc[index]['ebird_code'],  df.loc[index]['filename'].split('.')[0])

    librosa_audio, librosa_sample_rate = librosa.load(name)

    D = librosa.stft(librosa_audio)  # 

    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    plt.figure (figsize = (8,8))

    ### You can display the spectrogram using librosa.display.specshow 

    librosa.display.specshow(S_db)

    plt.title('linear_spectrogram')

    plt.colorbar()

    return linear_spectrogram

linear_spectrogram(train_df)