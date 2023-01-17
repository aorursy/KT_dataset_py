import numpy as np

import pandas as pd

import warnings

import random



import librosa

import librosa.display

from tqdm import tqdm_notebook as tqdm



import tarfile

from pathlib import Path



warnings.filterwarnings('ignore')
input_dir = Path('../input/birdsong-recognition/train_audio')
MFCC = {

    "sr": 22050, # sampling rate for loading audio

    "n_mfcc": 12 # number of MFCC features per frame that can fit in HDD

}
def load_audio(filename):

    try:

        return librosa.load(filename, sr=None)

    except Exception as e:

        print(f"Cannot load '{filename}': {e}")

        return None
def extract_mfcc(y, sr=22050, n_mfcc=10):

    try:

        return librosa.feature.mfcc(y=y, 

                                    sr=sr if sr > 0 else MFCC["sr"], 

                                    n_mfcc=n_mfcc)

    except Exception as e:

        print(f"Cannot extract MFCC: {e}")

        return None
def parse_audio(input_dir, output_file, max_per_label=10000):

    

    with tarfile.open(output_file, "w:xz") as tar:

    

        sub_dirs = list(input_dir.iterdir())    

        for sub_dir in tqdm(sub_dirs):



            for i, mp3 in enumerate(sub_dir.glob("*.mp3")):



                if i >= max_per_label:

                    break



                ysr = load_audio(mp3)

                if ysr is None:

                    continue



                mfcc = extract_mfcc(y=ysr[0], 

                                    sr=ysr[1], 

                                    n_mfcc=MFCC['n_mfcc'])

                if mfcc is None:

                    continue

                

                filename = Path(f"{mp3.name}.npy")

                np.save(filename, mfcc)            

                tar.add(filename)

                filename.unlink()
output_file = Path('train_features.xz')
parse_audio(input_dir, output_file)
sub_df = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

sub_df.to_csv('submission.csv', index = None)