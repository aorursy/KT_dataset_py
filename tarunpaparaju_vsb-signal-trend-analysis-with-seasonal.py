##### This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pyarrow.parquet as pq

import matplotlib.pyplot as plt

import time



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
!pip install --user seasonal

from seasonal import fit_seasons, adjust_seasons
praq = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(1000)]).to_pandas()

signals = praq.T.values.astype(float)

metadata = pd.read_csv('../input/metadata_train.csv', nrows=1000)

targets = metadata['target']
pos_indices = []

neg_indices = []

for i in range(len(targets)):

    if targets[i] == 0:

        neg_indices.append(i)

    else:

        pos_indices.append(i)



indices = [index for index in range(signals.shape[1]) if index % 20 == 0]

neg_signals = signals[neg_indices]

pos_signals = signals[pos_indices]
s = time.time()

for i in range(50):

    signal = neg_signals[i]

    short_signal = signal[indices]

    seasons, trend = fit_seasons(short_signal)

    e = time.time()

    print("SIGNAL SAMPLE {}".format(i+1))

    print("Total time : {}".format(str(e - s) + " s"))

    

    color = 'g'

    plt.plot(short_signal, color)

    plt.show()

    print("Trend")

    plt.plot(trend, color)

    plt.show()

    print("Noise")

    plt.plot(short_signal - trend, color)

    plt.show()
s = time.time()

for i in range(50):

    signal = pos_signals[i]

    short_signal = signal[indices]

    seasons, trend = fit_seasons(short_signal)

    e = time.time()

    print("SIGNAL SAMPLE {}".format(i+1))

    print("Total time : {}".format(str(e - s) + " s"))

    

    color = 'r'

    plt.plot(short_signal, color)

    plt.show()

    print("Trend")

    plt.plot(trend, color)

    plt.show()

    print("Noise")

    plt.plot(short_signal - trend, color)

    plt.show()