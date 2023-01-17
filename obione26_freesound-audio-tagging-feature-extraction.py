import os

import time

import librosa

import numpy as np

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from tqdm import tqdm

from functools import partial

import multiprocessing
!mkdir -p ../data/train/curated

!unzip -q /kaggle/input/freesound-audio-tagging-2019/train_curated.zip -d ../data/train/curated/wav



# Takes very long and a lot of memory

# !mkdir -p ../data/train/noisy

# !unzip -q /kaggle/input/freesound-audio-tagging-2019/train_noisy.zip -d ../data/train/noisy/wav



!mkdir -p ../data/test

!unzip -q /kaggle/input/freesound-audio-tagging-2019/test.zip -d ../data/test/wav
def load_train_data_df(mode='curated'):

    # Load training data filenames and labels (raw -> multilabels are represented as a string with comma separated values)

    data_folder = '/kaggle/input/freesound-audio-tagging-2019' 

    csv_path = f'{data_folder}/train_{mode}.csv'

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



def extract_mfcc(sample, n_mfcc=20, sr=44100):

    """ Return a matrix of shape (n_mfcc, int(seconds*sr/1024)). """

    mfccs = librosa.feature.mfcc(sample, sr=sr, n_mfcc=n_mfcc)

    return mfccs.astype(np.float32)



def extract_log_mel(sample, n_mels=128, sr=44100, hop=347):

    mel = librosa.feature.melspectrogram(sample, sr=sr, n_fft=20*n_mels, hop_length=hop, n_mels=n_mels, fmin=20, fmax=sr//2)

    logmel = librosa.core.power_to_db(mel, ref=1.0, amin=1e-10, top_db=None)

    return logmel.astype(np.float32)



def extract_features(path, save_folder='../data/train/curated', feat_type='logmel', n_feats=128, hop=347, sr=44100, save=False):

    start_time = time.time()



    sample, _ = librosa.load(path, sr=None)

    x, _ = librosa.effects.trim(sample)



    if feat_type == 'mfcc':

        x = extract_mfcc(x, n_feats, sr)

    elif feat_type == 'logmel':

        x = extract_log_mel(x, n_feats, sr, hop)

        

    if save:

        if not os.path.exists(f'{save_folder}/{feat_type}'):

            os.makedirs(f'{save_folder}/{feat_type}')

        filename = path.split('/')[-1].split('.')[0]

        np.save(f"{save_folder}/{feat_type}/{filename}.npy", x)

    

def extract_serie(wav_paths, save_folder='../data/train/curated', feat_type='logmel', n_feats=128, hop=347, sr=44100, save=False):

    start_time = time.time()

    

    with tqdm(total=len(curated_train_wav_paths)) as bar:

        for path in wav_paths:

            extract_features(path, save_folder, feat_type, n_feats, hop, sr, save)

            bar.update(1)

            

    print(f'Successfully extracted {feat_type} features in {save_folder} ! (took {time.time() - start_time:.2f}s).')

    

def extract_parallel(wav_paths, save_folder='../data/train/curated', feat_type='logmel', n_feats=128, hop=347, sr=44100, save=False):

    start_time = time.time()

    

    n_cores = multiprocessing.cpu_count()

    print(f'Extracting {feat_type} using {n_cores} cores...')

    

    with tqdm(total=len(wav_paths)) as bar, multiprocessing.Pool(processes=n_cores) as pool:

        function = partial(extract_features, save_folder=save_folder, feat_type=feat_type, n_feats=n_feats, hop=hop, sr=sr, save=save)

        for _ in pool.map(function, wav_paths):        

            bar.update(1)

        

    print(f'Successfully extracted {feat_type} features in {save_folder} ! (took {time.time() - start_time:.2f}s).')
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



use_parallel = True



if use_parallel:

    extract_parallel(curated_train_wav_paths, save_folder='../data/train/curated', save=True)

#     extract_parallel(noisy_train_wav_paths, save_folder='../data/train/noisy', save=True)

    extract_parallel(test_wav_paths, save_folder='../data/test', save=True)   

else:

    extract_serie(curated_train_wav_paths, save_folder='../data/train/curated', save=True)

#     extract_serie(noisy_train_wav_paths, save_folder='../data/train/noisy', save=True)

    extract_serie(test_wav_paths, save_folder='../data/test', save=True)
!zip -rq curated_train_logmel.zip ../data/train/curated/logmel

# !zip -r noisy_train_logmel.zip ../data/train/noisy/logmel

!zip -rq test_logmel.zip ../data/test/logmel
!du -m curated_train_logmel.zip

# !du -m noisy_train_logmel.zip

!du -m test_logmel.zip