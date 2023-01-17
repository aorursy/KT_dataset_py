# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        break

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data_path = os.path.join('/', 'kaggle', 'input', 'birdsong-recognition')
fnames = os.listdir(data_path)
fpaths = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
fname_dict = dict(zip(fnames, fpaths)) # convenient mapping between filenames and absolute path
train_data = pd.read_csv(fname_dict['train.csv'])
distilled_data = train_data[(train_data.rating >= 4.0)
                            & (train_data.secondary_labels == "[]")
                            & (train_data.background.isna())
                            & (train_data.type == "song")]
print(f"size full dataset: {len(train_data)}")
print(f"size distilled dataset: {len(distilled_data)}, keeping {round(len(distilled_data)/len(train_data)*100, 2)} %")
distilled_data.head()
import librosa
import warnings
from typing import NamedTuple, List
class Spectrogram(NamedTuple):
    ebird_code: str
    mel: np.array
    fpath: str
class BirdSpecs(NamedTuple):
    ebird_code: str
    spectrograms: List[Spectrogram]
def generate_mel_spectrogram(fpath):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        y, sr = librosa.load(fpath)
        return librosa.feature.melspectrogram(y=y, sr=sr)
    
    return None
train_audio_path = os.path.join(data_path, 'train_audio')
def get_fpath(ebird_code, filename):
    return os.path.join(train_audio_path, ebird_code, filename)
birds = dict()
for ix, row in distilled_data.reset_index().iterrows():
    print(f"Row: {ix}", end="\r")
    if row.ebird_code in birds:
        bird = birds[row.ebird_code] 
    else:
        bird = BirdSpecs(ebird_code=row.ebird_code,
                         spectrograms=list())
        
    fpath = get_fpath(row.ebird_code, row.filename)
    try:
        mel = generate_mel_spectrogram(fpath)
    
    except BaseException:
        mel = None
    spec = Spectrogram(ebird_code=bird.ebird_code,
                       mel=mel,
                       fpath=fpath)
    bird.spectrograms.append(spec)
    birds[row.ebird_code] = bird
birds