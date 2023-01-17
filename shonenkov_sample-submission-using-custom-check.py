import pandas as pd

import numpy as np

from glob import glob

import random

import os

import time

import torch

import torchaudio

import librosa
BASE_TEST_DIR = '../input/birdsong-recognition' if os.path.exists('../input/birdsong-recognition/test_audio') else '../input/birdcall-check'
df_test = pd.read_csv(f'{BASE_TEST_DIR}/test.csv')

df_train = pd.read_csv('../input/birdsong-recognition/train.csv')

all_birds = df_train['ebird_code'].unique()
def random_predict():

    birds = random.choices(all_birds, k=random.randint(0,2)) or ['nocall']

    return ' '.join(birds)
sub_test_12 = df_test[df_test.site.isin(['site_1', 'site_2'])]

sub_test_3 = df_test[df_test.site.isin(['site_3'])]
TEST_FOLDER = f'{BASE_TEST_DIR}/test_audio'



def custom_read_audio(audio_path, sr=44100):

    """

    author: @shonenkov 

    

    Super fast method, without exceptions. 

    return waveform <torch.tensor>, sample_rate <number>

    """

    try:

        waveform, sample_rate = torchaudio.load(audio_path, normalization=True)

        if sample_rate != sr:

            waveform = torchaudio.transforms.Resample(sample_rate, sr)(waveform)

            sample_rate = sr

    except RuntimeError:

        waveform, sample_rate = librosa.load(audio_path, sr=sr, mono=False)

        waveform = torch.from_numpy(waveform)

        if waveform.shape[0] not in [1, 2]:

            waveform = waveform.unsqueeze(0)

    return waveform, sample_rate
submission = {'row_id': [], 'birds': []}



for audio_id, data in sub_test_12.groupby('audio_id'):

    waveform, sample_rate = custom_read_audio(f'{TEST_FOLDER}/{audio_id}.mp3')

    submission['row_id'].extend(data['row_id'].values)

    submission['birds'].extend([random_predict() for i in range(data.shape[0])])



for _, row in sub_test_3.iterrows():

    row_id, audio_id = row['row_id'], row['audio_id']

    waveform, sample_rate = custom_read_audio(f'{TEST_FOLDER}/{audio_id}.mp3')

    submission['row_id'].append(row_id)

    submission['birds'].append(random_predict())



submission = pd.DataFrame(submission)

submission.head()
submission.to_csv('submission.csv', index=False)