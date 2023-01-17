import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import torch

import torchaudio

import matplotlib.pyplot as plt



import IPython.display as ipd







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from os import path







%matplotlib inline
!pip install tinytag
DATA_PATH = '../input/birdsong-recognition/'

AUDIO_PATH = "../input/birdsong-recognition/train_audio"
display(ipd.Audio('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3'))
display(ipd.Audio('../input/birdsong-recognition/train_audio/ameavo/XC133080.mp3'))
display(ipd.Audio('../input/birdsong-recognition/train_audio/amebit/XC127371.mp3'))
df_train = pd.read_csv(DATA_PATH + 'train.csv')
df_train.head()
species_count = df_train['ebird_code'].value_counts(sort=True)

species_count = species_count[250:]

plt.figure(figsize=(15,7))

sns.barplot(species_count.index,species_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Birds species distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Birds code', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.ebird_code.value_counts()))

print('Maximum samples per category = ', max(df_train.ebird_code.value_counts()))
print("Total number of labels: ",len(df_train['ebird_code'].value_counts()))

print("Labels: ", df_train['ebird_code'].unique())
from wordcloud import WordCloud

wordcloud = WordCloud(background_color="white", max_font_size=50, width=800, height=500, collocations=False, max_words=200).generate(' '.join(df_train['ebird_code']))

plt.figure(figsize=(18,10))

plt.imshow(wordcloud, cmap=plt.cm.gray)

plt.title("Wordcloud for Labels in Train Curated", fontsize=25)

plt.axis("off")

plt.show()
playback_used_count  = df_train['playback_used'].value_counts(sort=True)

playback_used_count = playback_used_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(playback_used_count.index,playback_used_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Playback_used distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Playback_used', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.playback_used.value_counts()))

print('Maximum samples per category = ', max(df_train.playback_used.value_counts()))
channels_count  = df_train['channels'].value_counts(sort=True)

channels_count = channels_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(channels_count.index,channels_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Channels distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Channels', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.channels.value_counts()))

print('Maximum samples per category = ', max(df_train.channels.value_counts()))
date_count  = df_train['date'].value_counts(sort=True)

date_count = date_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(date_count.index,date_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Date distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.date.value_counts()))

print('Maximum samples per category = ', max(df_train.date.value_counts()))
pitch_count  = df_train['pitch'].value_counts(sort=True)

pitch_count = pitch_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(pitch_count.index,pitch_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Pitch distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Pitch', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.pitch.value_counts()))

print('Maximum samples per category = ', max(df_train.pitch.value_counts()))
duration_count  = df_train['duration'].value_counts(sort=True)

duration_count = duration_count[:30]

plt.figure(figsize=(15,7))

sns.barplot(duration_count.index,duration_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Duration distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Duration', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.duration.value_counts()))

print('Maximum samples per category = ', max(df_train.duration.value_counts()))
speed_count  = df_train['speed'].value_counts(sort=True)

speed_count = speed_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(speed_count.index,speed_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Speed distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Speed', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.speed.value_counts()))

print('Maximum samples per category = ', max(df_train.speed.value_counts()))
type_count  = df_train['type'].value_counts(sort=True)

type_count = type_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(type_count.index,type_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Type distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Type', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.type.value_counts()))

print('Maximum samples per category = ', max(df_train.type.value_counts()))
sampling_rate_count  = df_train['sampling_rate'].value_counts(sort=True)

plt.figure(figsize=(15,7))

sns.barplot(sampling_rate_count.index,sampling_rate_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Sampling rate distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Rate (Hz)', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.sampling_rate.value_counts()))

print('Maximum samples per category = ', max(df_train.sampling_rate.value_counts()))
elevation_count  = df_train['elevation'].value_counts(sort=True)

elevation_count = elevation_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(elevation_count.index,elevation_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Elevation distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Elevation', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.elevation.value_counts()))

print('Maximum samples per category = ', max(df_train.elevation.value_counts()))
bitrate_of_mp3_count  = df_train['bitrate_of_mp3'].value_counts(sort=True)

bitrate_of_mp3_count = bitrate_of_mp3_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(bitrate_of_mp3_count.index,bitrate_of_mp3_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Bitrate_of_mp3 distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Bitrate_of_mp3', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.bitrate_of_mp3.value_counts()))

print('Maximum samples per category = ', max(df_train.bitrate_of_mp3.value_counts()))
country_count  = df_train['country'].value_counts(sort=True)

country_count = country_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(country_count.index,country_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Countries distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Country', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.country.value_counts()))

print('Maximum samples per category = ', max(df_train.country.value_counts()))
recordist_count  = df_train['recordist'].value_counts(sort=True)

recordist_count = recordist_count[:10]

plt.figure(figsize=(15,7))

sns.barplot(recordist_count.index,recordist_count.values,palette=("Blues_d"), alpha=0.85)

plt.title('Recordists distribution', fontsize=18)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Recordists', fontsize=12)

plt.show()
print('Minimum samples per category = ', min(df_train.recordist.value_counts()))

print('Maximum samples per category = ', max(df_train.recordist.value_counts()))
df_test = pd.read_csv(DATA_PATH + 'test.csv')
df_test.head()
ex_test_summary = pd.read_csv(DATA_PATH + 'example_test_audio_summary.csv')
ex_test_summary.head()
ex_test_meta = pd.read_csv(DATA_PATH + 'example_test_audio_metadata.csv')
ex_test_meta.head()
files = folders = 0

FolderList = []

for _, dirnames, filenames in os.walk(AUDIO_PATH):

    files += len(filenames)

    folders += len(dirnames)

    FolderList.append(dirnames)

        

print("There are {:,} files, and {:,} folders".format(files, folders))
from tinytag import TinyTag
tag = TinyTag.get(AUDIO_PATH + "/aldfly/XC134874.mp3", image=True)
print('album:',tag.album)

print('album artist:',tag.albumartist) # album artist

print('artist name:',tag.artist)       # artist name

print('number of bytes before audio data begins:',tag.audio_offset)  # number of bytes before audio data begins

print('bitrate in kBits/s:',tag.bitrate) # bitrate in kBits/s

print('file comment', tag.comment)     # file comment

print('composer', tag.composer)        # composer

print('disc number', tag.disc)         # disc number

print('total number of discs',tag.disc_total)    # total number of discs

print('duration of the song in seconds', tag.duration)      # duration of the song in seconds

print('file size in bytes', tag.filesize)      # file size in bytes

print('genre', tag.genre)              # genre

print('samples per second', tag.samplerate)    # samples per second

print('title of the song', tag.title)  # title of the song

print('track number', tag.track)       # track number

print('total number of tracks', tag.track_total)     # total number of tracks

print('year or data', tag.year)        # year or data
from IPython.display import Audio

from matplotlib import pyplot as plt

import torchaudio

import torch



import librosa

import librosa.display
%%time

filename = "../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3"



waveform, sample_rate = torchaudio.load(filename)



plt.plot(waveform.t().numpy())
%%time

x,sr = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3')

librosa.display.waveplot(x, sr=sr)
librosa.feature.melspectrogram(x, sr=sample_rate, n_mels=128)
%%time

fig, ax = plt.subplots(1, 2, figsize=(16, 12))



ax[0].imshow(torchaudio.transforms.Spectrogram(n_fft=2000)(waveform).log2()[0,:,:].numpy(), cmap='ocean');

ax[0].set_title('Spectrogram');

ax[1].imshow(torchaudio.transforms.MelSpectrogram(n_fft=16000)(waveform).log2()[0].numpy(), cmap='inferno');

ax[1].set_title('MelSpectrogram');
%%time

X = librosa.stft(x)

Xdb = librosa.amplitude_to_db(abs(X))

plt.figure(figsize=(14, 5))

librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')

plt.colorbar()