import h5py

import numpy as np

import IPython

import IPython.display as ipd

import librosa

import librosa.display



def load_h5(filenames, h5_filename):

    if not isinstance(filenames, list):

        filenames = [filenames]

    with h5py.File(h5_filename, mode='r') as dataset:

        samples = [ dataset[f][()] for f in filenames ]

        return samples



def listen_sample(sample, sr=44100):

    return IPython.display.display(ipd.Audio(data=sample, rate=sr))

    

dataset_file = '/kaggle/input/fat2019wav/train_curated_wav.h5'

    

with h5py.File(dataset_file, mode='r') as dataset:

    samples_list = list(dataset.keys())
%%time

samples = load_h5(samples_list[:10], dataset_file)

listen_sample(samples[0])