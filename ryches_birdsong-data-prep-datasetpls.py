import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
base_path = "../input/birdsong-recognition/"

train_df = pd.read_csv(os.path.join(base_path, "train.csv"))

test_df = pd.read_csv(os.path.join(base_path, "test.csv"))

os.listdir("../input/birdsong-recognition/")
train_df["filename"]
#preview of an audio clip

import IPython

IPython.display.Audio(base_path + "train_audio/" + train_df["ebird_code"].iloc[0] +"/"+ train_df["filename"].iloc[0])
import librosa

import matplotlib.pyplot as plt

from scipy import signal

y, sr = librosa.load(base_path + "train_audio/" + train_df["ebird_code"].iloc[0] +"/"+ train_df["filename"].iloc[0])

M = librosa.feature.melspectrogram(y=y, sr = 48000)

M = librosa.power_to_db(M, ref=np.max)
M.shape
plt.imshow(M)
import cv2
from zipfile import ZipFile
#from https://www.kaggle.com/daisukelab/creating-fat2019-preprocessed-data

def mono_to_color(X, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6):

    # Stack X as [X,X,X]

#     X = np.stack([X, X, X], axis=-1)



    # Standardize

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

        V = 255 * (V - norm_min) / (norm_max - norm_min)

        V = V.astype(np.uint8)

    else:

        # Just zero

        V = np.zeros_like(Xstd, dtype=np.uint8)

    return V
def build_spectogram(path, sr, duration):

    try:

        y, _ = librosa.load(path, sr = sr, duration=duration)

        M = librosa.feature.melspectrogram(y=y, sr=sr)

        M = librosa.power_to_db(M)

        M = mono_to_color(M)

        cv2.imwrite(path.split("/")[-1][:-4] + ".jpg", M, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        with ZipFile('birdsongs.zip', 'a') as myzip:

            myzip.write(path.split("/")[-1][:-4] + ".jpg")

        os.remove(path.split("/")[-1][:-4] + ".jpg")

    except:

        print("spectrogram generation failed")
train_df["duration"].hist(bins = 100)
import warnings

from tqdm import tqdm_notebook as tqdm
!rm -rf birdsongs.zip
for i in tqdm(range(len(train_df))):

    warnings.simplefilter("ignore")

    row = train_df.iloc[i]

    duration = np.min([400, row["duration"]])

    if duration != row["duration"]:

        print("truncated audio")

    fp = base_path + "train_audio/" + row["ebird_code"] +"/"+ row["filename"]

    sr = float(row["sampling_rate"].split(" ")[0])

    build_spectogram(fp, sr, duration)
!ls -l
train_df.iloc[-1]
fp