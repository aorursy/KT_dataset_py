import os

import time

import h5py

import tables

import librosa

import numpy as np

import pandas as pd

import IPython

import IPython.display as ipd

from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm_notebook

import tensorflow as tf
# Helper functions

def load_train_data_df(mode='curated'):

    # Load training data filenames and labels (raw -> multilabels are represented as a string with comma separated values)

    csv_path = f'/kaggle/input/freesound-audio-tagging-2019/train_{mode}.csv'

    raw_df = pd.read_csv(csv_path, index_col='fname')

        

    # Extract list of expected labels

    sub = pd.read_csv('/kaggle/input/freesound-audio-tagging-2019/sample_submission.csv', index_col='fname')

    labels_list = sub.columns.values 



    # Encode multi-labels in a binary vector

    splitted_labels = [ labels.split(',') for labels in raw_df['labels'].values ]

    encoder = MultiLabelBinarizer()

    encoded_labels = encoder.fit_transform(splitted_labels)



    # Create a new pandas Dataframe to represent training labels as binary vectors

    labels_df = pd.DataFrame(data=encoded_labels, index=list(raw_df.index), columns=labels_list)

    

    return labels_df



def listen_sample(sample, sr=44100):

    return IPython.display.display(ipd.Audio(data=sample, rate=sr))



def load_sample(path, trim=True):

    input_data = tf.io.read_file(path)

    sample, _ = tf.audio.decode_wav(input_data)

    sample = sample.numpy().flatten()

    if trim:

        sample , _ = librosa.effects.trim(sample)

    return sample



def compress(x, eps=1e-6):

    x_min = np.min(x, keepdims=True)

    x_max = np.max(x, keepdims=True)

    x = 255 * (x - x_min)/(x_max - x_min + eps)

    return x.astype(np.uint8)



def save_h5(savepath, X):

    with tables.open_file(savepath, mode='w') as h5_file:

        filters = tables.Filters(complib='zlib', complevel=1)

        for filename, x in X.items():

            h5_file.create_carray('/', f"t{filename.split('.')[0]}", obj=x, filters=filters)



def load_h5(filenames, h5_filename):

    if not isinstance(filenames, list):

        filenames = [filenames]

    with h5py.File(h5_filename, mode='r') as dataset:

        samples = [ dataset[f][()] for f in filenames ]

        return samples



def save_features(wav_paths, savepath='train_curated_wav', format='pkl', n_splits=1):

    """ Save compressed features (wav) into the specified format. Increase n_splits if you go out of RAM. 

    In terms of compression ratio: h5 (best) > pkl > npy.

    In terms of read speed: pkl (best) > npy > h5.

    """

    assert format in ['pkl', 'npy', 'h5'], 'Wrong format argument !'

    

    start_time = time.time()



    for split_idx, split_paths in enumerate(np.array_split(wav_paths, n_splits)):

        X = {}

        for path in split_paths:

            filename = path.split('/')[-1]

            x = load_sample(path, trim=True)

            x = compress(x)

            X[filename] = x



        savepath_split = savepath + ('', f'_{split_idx + 1}')[n_splits > 1]

        if format == 'pkl':

            save_pkl(savepath_split + '.pkl', X)

        elif format == 'h5':

            save_h5(savepath_split + '.h5', X)

        elif format == 'npy':

            save_npy(savepath_split, X)



        print(f'Successfully saved .wav features in {savepath_split} ! (took {time.time() - start_time:.2f}s).')
%%time



# Unzip all files (~8mn)

!mkdir -p ../data/train/curated

!mkdir -p ../data/train/noisy

!mkdir -p ../data/test



!unzip -q /kaggle/input/freesound-audio-tagging-2019/train_curated.zip -d ../data/train/curated/wav 

!unzip -q /kaggle/input/freesound-audio-tagging-2019/train_noisy.zip -d ../data/train/noisy/wav

!unzip -q /kaggle/input/freesound-audio-tagging-2019/test.zip -d ../data/test/wav
# Load data filenames and labels

curated_train_labels = load_train_data_df(mode='curated')

noisy_train_labels = load_train_data_df(mode='noisy')

test_labels = pd.read_csv('/kaggle/input/freesound-audio-tagging-2019/sample_submission.csv', index_col='fname')



# Main info about the training/testing sets

print(f'{curated_train_labels.shape[1]} possible classes.')

print(f'{curated_train_labels.shape[0]} curated training samples.')

print(f'{noisy_train_labels.shape[0]} noisy training samples.')

print(f'{test_labels.shape[0]} test samples.')



curated_train_wav_paths = '../data/train/curated/wav/' + curated_train_labels.index.values

noisy_train_wav_paths = '../data/train/noisy/wav/' + noisy_train_labels.index.values

test_wav_paths = '../data/test/wav/' + test_labels.index.values
%%time

save_features(curated_train_wav_paths, savepath='../train_curated_wav', format='h5', n_splits=1) # 2mn

save_features(noisy_train_wav_paths, savepath='../train_noisy_wav', format='h5', n_splits=2) # 20mn

save_features(test_wav_paths, savepath='../test_wav', format='h5', n_splits=1) # 2mn