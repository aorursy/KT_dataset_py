# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Other  

import librosa

import librosa.display

import json

import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf

from matplotlib.pyplot import specgram

import pandas as pd

import seaborn as sns

import glob 

import os

from tqdm import tqdm

import pickle

import IPython.display as ipd  # To play sound in the notebook
# Use one audio file in previous parts again

fname = '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/disco/disco.00072.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Paly it again to refresh our memory

ipd.Audio(data, rate=sampling_rate)
# Use one audio file in previous parts again

fname = '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/country/country.00044.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Paly it again to refresh our memory

ipd.Audio(data, rate=sampling_rate)
# Use one audio file in previous parts again

fname = '/kaggle/input/gtzan-dataset-music-genre-classification/Data/genres_original/classical/classical.00004.wav'  

data, sampling_rate = librosa.load(fname)

plt.figure(figsize=(15, 5))

librosa.display.waveplot(data, sr=sampling_rate)



# Paly it again to refresh our memory

ipd.Audio(data, rate=sampling_rate)