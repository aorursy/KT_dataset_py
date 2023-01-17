import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import tensorflow as tf

import librosa

import keras

import matplotlib.pyplot as plt

import scipy.io

from scipy.fftpack import rfft

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import MultiLabelBinarizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, accuracy_score

from tqdm import tqdm, tqdm_notebook; tqdm.pandas() # Progress bar

print(os.listdir("../input"))
bad_clips = ["f76181c4.wav", "77b925c2.wav", "6a1f682a.wav", "c7db12aa.wav", "7752cc8a.wav"]

df = pd.read_csv("../input/train_curated.csv")

test = pd.read_csv("../input/sample_submission.csv")

df = df[~df.fname.isin(bad_clips)]

df = df.groupby('labels').head(10)



#y = df.labels.apply(lambda x: x.split(","))

#mlb = MultiLabelBinarizer()

#yc = mlb.fit_transform(y)

for c in test.columns[1:]:

    df[c] = df.labels.str.contains(c)

mels = []

for f in tqdm_notebook(df.fname):

    #y, sr = librosa.load("../input/train_curated/" + f, sr=None)

    sr, y = scipy.io.wavfile.read("../input/train_curated/" + f)

    y = y.astype(float)

    y, index = librosa.effects.trim(y)

    y = librosa.util.fix_length(y, sr * 5)

    mel = librosa.feature.mfcc(y=y, sr=sr)

    mels.append(mel.T)
mels[0].shape
x = np.vstack(mels)

print(x.shape)



y = df.drop(['fname', 'labels'], axis=1)

y = np.repeat(y.as_matrix(), len(mels[0]), axis=0)

print(x.shape, y.shape)

xtrain, xtest, ytrain, ytest = train_test_split(x, y)



model = RandomForestClassifier(verbose=0)

model.fit(xtrain, ytrain)
model.score(xtest,ytest)
testmels = []

for f in tqdm_notebook(test.fname):

    #y, sr = librosa.load("../input/test/" + f, sr=None)

    sr, y = scipy.io.wavfile.read("../input/test/" + f)

    y = y.astype(float)

    y, index = librosa.effects.trim(y)

    y = librosa.util.fix_length(y, sr)

    mel = librosa.feature.mfcc(y=y, sr=sr)

    testmels.append(mel.T)
%%timeit

for i in range(len(test)):

    preds = np.mean(model.predict(testmels[i]), axis=0)

    test.iloc[i, 1:] = preds
test.to_csv("submission.csv", index=False)