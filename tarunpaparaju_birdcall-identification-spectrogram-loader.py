!pip install -q pydub
import os

import gc

import cv2

import time

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

from joblib import Parallel, delayed



import warnings

warnings.filterwarnings('ignore')



import pydub

import librosa

import librosa.display

from pydub import AudioSegment as AS

from librosa.feature import melspectrogram

from librosa.core import power_to_db as ptdb



from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences as pad
N = 8

SR = 44100

CHUNKS = 1

TSR = 32000

N_MELS = 128

POP_FRAC = 0.25

MAXLEN = 1000000

AMPLITUDE = 1000

CHUNK_SIZE = 500000
os.listdir('../input')
TEST_DATA_PATH = '../input/birdsong-recognition/test.csv'

TRAIN_DATA_PATH = '../input/birdsong-recognition/train.csv'

TEST_AUDIO_PATH = '../input/birdsong-recognition/test_audio/'

TRAIN_AUDIO_PATH = '../input/birdsong-recognition/train_audio/'

CHECKING_PATH = '../input/prepare-check-dataset/birdcall-check/'
sub = os.path.exists(TEST_AUDIO_PATH)

TEST_DATA_PATH = TEST_DATA_PATH if sub else CHECKING_PATH + 'test.csv'

TEST_AUDIO_PATH = TEST_AUDIO_PATH if sub else CHECKING_PATH + 'test_audio/'
test_df = pd.read_csv(TEST_DATA_PATH)

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df.head()
train_df.head()
keys = set(train_df.ebird_code)

values = np.arange(0, len(keys))

code_dict = dict(zip(sorted(keys), values))
def normalize(x):

    return np.float32(x)/2**15



def read(file, norm=False):

    try:

        a = AS.from_mp3(file)

        a = a.set_frame_rate(TSR)

    except:

        return TSR, np.zeros(MAXLEN)



    y = np.array(a.get_array_of_samples())

    if a.channels == 2: y = y.reshape((-1, 2))

    if norm: return a.frame_rate, normalize(y)

    if not norm: return a.frame_rate, np.float32(y)



def write(file, sr, x, normalized=False):

    birds_audio_bitrate, file_format = '320k', 'mp3'

    ch = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1

    y = np.int16(x * 2 ** 15) if normalized else np.int16(x)

    song = AS(y.tobytes(), frame_rate=sr, sample_width=2, channels=ch)

    song.export(file, format=file_format, bitrate=birds_audio_bitrate)
def get_idx(length):

    length = get_len(length)

    max_idx = MAXLEN - CHUNK_SIZE

    idx = np.random.randint(length + 1)

    chunk_range = idx, idx + CHUNK_SIZE

    chunk_idx = max([0, chunk_range[0]])

    chunk_idx = min([chunk_range[1], max_idx])

    return (chunk_idx, chunk_idx + CHUNK_SIZE)



def get_len(length):

    if length > MAXLEN: return MAXLEN

    if length <= MAXLEN: return int(length*POP_FRAC)
def get_chunk(data, length):

    index = get_idx(length)

    return data[index[0]:index[1]]



def get_signal(data):

    length = max(data.shape)

    data = data.T.flatten().reshape(1, -1)

    data = np.float32(pad(data, maxlen=MAXLEN).reshape(-1))

    return [get_chunk(data, length) for _ in range(CHUNKS)]
def to_imagenet(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    mean = mean or X.mean()

    X = X - mean

    std = std or X.std()

    Xstd = X / (std + eps)

    _min, _max = Xstd.min(), Xstd.max()

    norm_max = norm_max or _max

    norm_min = norm_min or _min

    if (_max - _min) > eps:

        # Normalize to [0, 255]

        V = Xstd

        V[V < norm_min] = norm_min

        V[V > norm_max] = norm_max

        V = 255*((V - norm_min) / (norm_max - norm_min))

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return np.stack([V]*3, axis=-1)
def get_melsp(data):

    melsp = melspectrogram(data, n_mels=N_MELS)

    return to_imagenet(librosa.power_to_db(melsp))



def get_melsp_img(data):

    data = get_signal(data)

    return np.stack([get_melsp(point) for point in data])
def save(indices, path):

    folder = TRAIN_AUDIO_PATH



    for index in tqdm(indices):

        file_name = train_df.filename[index]

        ebird_code = train_df.ebird_code[index]



        default_signal = np.random.random(MAXLEN)*AMPLITUDE

        default_values = SR, np.int32(np.round(default_signal))



        values = read(folder + ebird_code + '/' + file_name)

        _, data = values if len(values) == 2 else default_values

        

        image = np.nan_to_num(get_melsp_img(data))[0]

        cv2.imwrite(path + file_name + '.jpg', image); del image; gc.collect()
train_ids = np.array_split(np.arange(len(train_df)), 5)

train_ids_1, train_ids_2, train_ids_3, train_ids_4, train_ids_5 = train_ids
train_ids_1 = np.array_split(np.array(train_ids_1), N)

train_ids_2 = np.array_split(np.array(train_ids_2), N)

train_ids_3 = np.array_split(np.array(train_ids_3), N)

train_ids_4 = np.array_split(np.array(train_ids_4), N)

train_ids_5 = np.array_split(np.array(train_ids_5), N)
!mkdir train_1

path = "train_1/"

parallel = Parallel(n_jobs=N, backend="threading")

parallel(delayed(save)(ids, path) for ids in train_ids_1)
!zip -r train_1.zip train_1

!rm -rf train_1
!mkdir train_2

path = "train_2/"

parallel = Parallel(n_jobs=N, backend="threading")

parallel(delayed(save)(ids, path) for ids in train_ids_2)
!zip -r train_2.zip train_2

!rm -rf train_2
!mkdir train_3

path = "train_3/"

parallel = Parallel(n_jobs=N, backend="threading")

parallel(delayed(save)(ids, path) for ids in train_ids_3)
!zip -r train_3.zip train_3

!rm -rf train_3
!mkdir train_4

path = "train_4/"

parallel = Parallel(n_jobs=N, backend="threading")

parallel(delayed(save)(ids, path) for ids in train_ids_4)
!zip -r train_4.zip train_4

!rm -rf train_4
!mkdir train_5

path = "train_5/"

parallel = Parallel(n_jobs=N, backend="threading")

parallel(delayed(save)(ids, path) for ids in train_ids_5)
!zip -r train_5.zip train_5

!rm -rf train_5