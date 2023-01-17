!conda install dask -y

import dask

import dask.array as da

import dask.bag as db

import dask.dataframe as dd

from dask.distributed import Client



!pip install pydub

from pydub import AudioSegment

from pydub.utils import mediainfo



!pip install zarr

from zarr import Zlib, BZ2, LZMA, Blosc



import librosa, os, time, gc, json

import numpy as np

import h5py

import pandas as pd

import matplotlib.pyplot as plt

import IPython.display as ipd

from skimage.util.shape import view_as_windows, view_as_blocks

import tensorflow as tf



pd.set_option("display.max_columns", 60)

pd.set_option("display.max_rows", 120)
temp_folder = '/home/dask_hdd'

os.mkdir(temp_folder)



client = Client(memory_limit='4GB', local_directory=temp_folder)

client
audio = da.from_zarr('/kaggle/input/bird-train')
audio
%%time

recording = audio[20000:20500].compute()
recording
train = pd.read_feather('/kaggle/input/bird-train/train.feather')

train = train[train['len']>0]



window_maxdim = np.prod(audio.shape[1:])

train['shape'] = train['len'].map(lambda x: (x+window_maxdim-x%window_maxdim)/window_maxdim)

train['slice'] = train['shape'].cumsum().astype(int)

train['slice'] = list(zip(train['slice'].shift(1,fill_value=0),train['slice']))



audio_index = train['slice'].map(lambda x: slice(*x))

train_index = train['slice'].apply(lambda x: np.arange(*x)).explode()

train_index = pd.Series(train_index.index, index=train_index.astype(int))
train.loc[5000]
audio_index.loc[5000]
recording = audio[audio_index.loc[5000]].compute()

print(recording.shape)
recording = recording.reshape(1,-1)

ipd.Audio(recording, rate=32000)
train.loc[5000].path
recording_original = AudioSegment.from_mp3(train.loc[5000].path).set_frame_rate(32000).set_channels(1)

recording_original
recording = recording.ravel()

recording_original = np.array(recording_original.get_array_of_samples(), dtype=np.int32)

np.all(np.equal(recording[-recording_original.shape[0]:], recording_original))
audio[99553]
train_index.loc[99553]
train.loc[train_index.loc[99553]]
%%time

audio.max().compute()
%%time

audio.argmax().compute()
%%time

np.unravel_index(1280089650, audio.shape)
audio[13151, 11, 13, 40].compute()
%%time

groupby_minmax = [dask.delayed(lambda x: (x.min(), x.max()))(audio[sli]) for sli in audio_index]
%%time

groupby_minmax = db.compute(groupby_minmax)
groupby_minmax