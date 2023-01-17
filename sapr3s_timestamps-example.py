import numpy as np

import pandas as pd

import IPython.display as ipd

import librosa

import warnings

warnings.filterwarnings('ignore')
input_path = '../input/birdsong-recognition/'

dataset_path = '../input/timestamps-for-train-data-cornell-birdcall/'

sr = 32000

pause_sec = 0.2

limit = True

limit_sec = 60
def get_aud_bird(df_full, bird, pause_sec):

    pause = np.zeros(int(pause_sec * sr))

    aud = pause



    df = df_full[df_full['bird']==bird]

    files = df['file'].unique()

    print(f'{bird} - {len(files)} files {len(df)} samples')

    

    for file in files:

        if limit:

            if len(aud) > limit_sec * sr:

                break

        aud_file , _ = librosa.load(f'{input_path}train_audio/{bird}/{file}', mono=True, sr=sr)

        aud_file = librosa.util.normalize(aud_file, axis=0)

        df_file = df[df['file']==file]

        for i, r in df_file.iterrows():

            start = int(r['start'] * sr)

            end = start + int(r['duration'] * sr)

            aud = np.concatenate((aud, aud_file[start:end], pause)) 

    

    return aud
df = pd.read_csv(f'{dataset_path}timestamps-train-cornell-birds.csv')
aud = get_aud_bird(df, 'aldfly', pause_sec)
ipd.Audio(aud, rate=sr)
aud_normoc = get_aud_bird(df, 'normoc', pause_sec)
ipd.Audio(aud_normoc, rate=sr)
aud_norcar = get_aud_bird(df, 'norcar', pause_sec)
ipd.Audio(aud_norcar, rate=sr)