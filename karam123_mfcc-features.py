import librosa,librosa.display

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

import numpy, scipy, matplotlib.pyplot as plt, IPython.display as ipd

#import stanford_mir; stanford_mir.init()

from ipywidgets import interact

from tqdm.notebook import tqdm

import glob

import joblib

import pandas as pd

import os

import tarfile

import numpy as np

from pathlib import Path
train = pd.read_csv('../input/birdsong-recognition/train.csv')

train.head()
train.shape
test = pd.read_csv('../input/birdsong-recognition/test.csv')

test.head()
test.shape
sub = pd.read_csv('../input/birdsong-recognition/sample_submission.csv')

sub.head()
test_audio = pd.read_csv('../input/birdsong-recognition/example_test_audio_summary.csv')

test_audio[80:100]
test_audio.shape
test_meta = pd.read_csv('../input/birdsong-recognition/example_test_audio_metadata.csv')

test_meta[123:145]
test_meta.shape
y,sr = librosa.load('../input/birdsong-recognition/train_audio/aldfly/XC134874.mp3')
# to display the audio

ipd.Audio(y, rate=sr)
# The sample rate is 22050 which means that the recorder was recording 22050 times per second.

print("sampling rate :",sr)
# The y.shape = (562011,) which means that there were 562011 samples recorded on just one channel (Mono) over the whole audio.

print(y.shape[0])
#Using simple math, you can calculate the duration of this audio file by dividing the total_number_of_samples over the sample_rate

print("duration of audio file :",y.shape[0]/sr)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

print(tempo)

print(beat_frames)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
beat_times
plt.figure(figsize=(14, 5))

librosa.display.waveplot(y, alpha=0.6)

plt.vlines(beat_times, -1, 1, color='r')

plt.ylim(-1, 1)
beat_times_diff = numpy.diff(beat_times)

plt.figure(figsize=(14, 5))

plt.hist(beat_times_diff, bins=50, range=(0,4))

plt.xlabel('Beat Length (seconds)')

plt.ylabel('Count')
hop_length = 512
22050/512
# Separate harmonics and percussives into two waveforms

y_harmonic, y_percussive = librosa.effects.hpss(y)
y_percussive.shape
#mel_feat = librosa.feature.melspectrogram(y=y, sr=sr)
#mel_feat.shape
mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=13)
mfcc.shape
(25.488*22050)/512
'''train_dir = '../input/birdsong-recognition/train_audio'''
'''diff_birds = os.listdir(train_dir)'''
'''len(diff_birds)'''
'''diff_birds[0]'''
'''list_mfcc = []

y_label = []'''
#hop_length = 1024
'''def calculate_mfcc(audio):

    # Load the example clip

    y, sr = librosa.load(audio)

    

    # Compute MFCC features from the raw signal

    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length, n_mfcc=10)

    return mfcc'''
'''n_jobs=4

verbose=1'''
'''for bird in tqdm(diff_birds):

    filelist = glob.glob(os.path.join(train_dir + '/' + bird,'*.mp3'))

    y_label.append([bird]*len(filelist))

    #for audio in filelist:

    mfcc_feature = [joblib.delayed(calculate_mfcc)(audio) for audio in filelist ]

    out = joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(mfcc_feature)

    list_mfcc.append(out)'''
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

            print(sub_dir)



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

                print(filename)

                np.save(filename, mfcc)            

                tar.add(filename)

                filename.unlink()
input_dir = Path('../input/birdsong-recognition/train_audio')
output_file = Path('train_features.xz')

output_file
input_dir = Path('../input/birdsong-recognition/train_audio')
MFCC = {

    "sr": 22050, # sampling rate for loading audio

    "n_mfcc": 12 # number of MFCC features per frame that can fit in HDD

}
parse_audio(input_dir, output_file)