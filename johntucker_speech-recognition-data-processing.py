# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Custom imports
import pickle

from pathlib import Path
from subprocess import check_output

from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# folder names are labels. File names are not necessarily unique without considering the label connected to the filename,
# therefore full paths including sorting folders are required.

folders = os.listdir("../input/tensorflow-speech-recognition-challenge/train/audio")
print(folders)
# Open test / validation lists
test_list = open("../input/tensorflow-speech-recognition-challenge/train/testing_list.txt", "r").readlines()
validation_list = open("../input/tensorflow-speech-recognition-challenge/train/validation_list.txt", "r").readlines()

# The contest does not inlude all labels, most are classified as "unknown"
contest_dict = {'yes': 0,
                'no': 1,
                'up': 2,
                'down': 3,
                'left': 4,
                'right': 5,
                'on': 6,
                'off': 7,
                'stop': 8,
                'go': 9,
                'unknown': 10,
                'silence': 11
               }
train_audio_path = '../input/tensorflow-speech-recognition-challenge/train/audio'

train_labels = os.listdir(train_audio_path)
print(f'Number of labels: {len(train_labels)}')

wavs = []
labels = []

# create a list of all the wav files and their labels which is NOT background noise
for label in train_labels:
    if label == '_background_noise_':
        continue
    files = os.listdir(train_audio_path + '/' + label)
    for f in files:
        if not f.endswith('wav'):
            continue
        wavs.append(f)
        labels.append(label)

# append on a list of generated background noise from the silence kernel to be included in training data
files = os.listdir('../input/silence')
for f in files:
    if not f.endswith('wav'):
        continue
    wavs.append(f)
    labels.append('silence')

x_train = []
x_val = []
x_test = []
y_train = []
y_val = []
y_test = []

# sort by comparing path to list, anything not found on the lists will be used as training data
for i in range(len(wavs)):
    if any(labels[i] + '/' + wavs[i] in s for s in test_list):
        x_test.append(wavs[i])
        y_test.append(labels[i])
    elif any(labels[i] + '/' + wavs[i] in s for s in validation_list):
        x_val.append(wavs[i])
        y_val.append(labels[i])
    else:
        x_train.append(wavs[i])
        y_train.append(labels[i])

# format as full file path, this will be useful when using a generator to train
x_train = ["{}/{}".format(y_train,x_train) for x_train, y_train in zip(x_train, y_train)]
x_val = ["{}/{}".format(y_val,x_val) for x_val, y_val in zip(x_val, y_val)]
x_test = ["{}/{}".format(y_test,x_test) for x_test, y_test in zip(x_test, y_test)]

# overwrite labels which are not present in the contest dictionary with the string 'unknown'
for i in range(len(y_train)):
    if not(y_train[i] in contest_dict):
        y_train[i] = 'unknown'

for i in range(len(y_val)):
    if not(y_val[i] in contest_dict):
        y_val[i] = 'unknown'

for i in range(len(y_test)):
    if not(y_test[i] in contest_dict):
        y_test[i] = 'unknown'

train_sequences = []
test_sequences = []

# create a list of numeric identifiers for use with NN when feeding dictionaries
for i in range(len(y_train)):
    train_sequences.append(contest_dict[y_train[i]])

for i in range(len(y_val)):
    train_sequences.append(contest_dict[y_val[i]])

for i in range(len(y_test)):
    test_sequences.append(contest_dict[y_test[i]])

label_list = x_train + x_val

# create label dictionaries
labels = dict(zip(label_list, train_sequences))
test_labels = dict(zip(x_test, test_sequences))

# create test, train, and validation dictionaries for training and final evaluation
test_dict = {'test': x_test}

partition = {'train': x_train,
             'validation': x_val}

# pickle the results
with open('SavedTestDict.pickle', 'wb') as handle:
    pickle.dump(test_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SavedPartition.pickle', 'wb') as handle:
    pickle.dump(partition, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SavedLabels.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('SavedTestLabels.pickle', 'wb') as handle:
    pickle.dump(test_labels, handle, protocol=pickle.HIGHEST_PROTOCOL)