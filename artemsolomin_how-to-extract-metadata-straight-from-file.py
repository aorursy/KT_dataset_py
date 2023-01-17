!pip install tinytag
!pip install mutagen
import numpy as np

import pandas as pd



import mutagen

from mutagen.mp3 import MP3

from tinytag import TinyTag





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



DATA_PATH = '../input/birdsong-recognition/'

AUDIO_PATH = "../input/birdsong-recognition/train_audio"



DATA_PATH_EXT_A_M = '../input/xeno-canto-bird-recordings-extended-a-m/'

AUDIO_PATH_EXT_A_M = "../input/xeno-canto-bird-recordings-extended-a-m/A-M"



DATA_PATH_EXT_N_Z = '../input/xeno-canto-bird-recordings-extended-n-z/'

AUDIO_PATH_EXT_N_Z = '../input/xeno-canto-bird-recordings-extended-n-z/N-Z'
df_train = pd.read_csv(DATA_PATH + 'train.csv')

df_train_ext_a_m = pd.read_csv(DATA_PATH_EXT_A_M + 'train_extended.csv')

df_train_ext_n_z = pd.read_csv(DATA_PATH_EXT_N_Z + 'train_extended.csv')
#Original dataset

for row in df_train.columns:

    print(row)
#New dataset from A to M

for row in df_train_ext_a_m.columns:

    print(row)
#New dataset from N to Z

for row in df_train_ext_n_z.columns:

    print(row)
a = df_train.columns

b = df_train_ext_a_m.columns

c = df_train_ext_n_z.columns
set(a).difference(b)
#Just to make sure that there is no difference between A-M and N-Z parts

set(b).difference(c)
#Common info

mutagen.File("../input/xeno-canto-bird-recordings-extended-a-m/A-M/aldfly/XC133197.mp3")
#Only what we need

audio = MP3("../input/xeno-canto-bird-recordings-extended-a-m/A-M/aldfly/XC133197.mp3")

print(audio.info.bitrate)

print(audio.info.length)

print(audio.info.channels)
tag = TinyTag.get("../input/xeno-canto-bird-recordings-extended-a-m/A-M/aldfly/XC133197.mp3", image=True)
print('file comment', tag.comment)

print('samples per second', tag.samplerate)

print('title of the song', tag.title)
df_train['volume'].value_counts()
df_train['number_of_notes'].value_counts()
df_train['pitch'].value_counts()
df_train['rating'].value_counts()
df_train['speed'].value_counts()