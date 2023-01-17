# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from scipy.io import wavfile

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

# importing required modules 

train_curated=pd.read_csv("../input/train_curated.csv")

train_curated.describe()
test=pd.read_csv("../input/sample_submission.csv")

test.describe()
test.head()
train_curated.head()
print(os.listdir("../input/train_curated/")[:20])
train_noisy=pd.read_csv('../input/train_noisy.csv')

train_noisy.head()
fname='../input/train_curated/0019ef41.wav'

rate, data = wavfile.read(fname)

print("Sampling (frame) rate = ", rate)

print("Total samples (frames) = ", data.shape)

print(data)
plt.plot(data, '-', );
plt.figure(figsize=(16, 4))

plt.plot(data[:500], '.'); plt.plot(data[:500], '-');
train_curated['nframes'] = train_curated['fname'].apply(lambda f: wavfile.read('../input/train_curated/' + f)[1].shape[0])
test['nframes'] = test['fname'].apply(lambda f: wavfile.read('../input/test/' + f)[1].shape[0])
train_curated.head()
test.head()