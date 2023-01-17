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
# import libraries

import time

import warnings

import numpy as np

import pandas as pd

import seaborn as sns

from scipy import stats

import matplotlib.pyplot as plt

from scipy.signal import hilbert

from scipy.signal import spectrogram

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

%matplotlib inline

sns.set()
# check submission file

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

submission.head()
PATH = "../input"

print("There are {} files in the test folder".format(len(os.listdir(os.path.join(PATH, "test")))))

# example 30 test files

tests = os.listdir(os.path.join(PATH, "test"))

n_tests = len(tests)



# downsampling size

ds = 200



# random sampling for test files

N = 30

np.random.seed(1220)

idx = np.random.choice(n_tests, N)



# fig & ax

nrow = int(N/5)

fig, ax = plt.subplots(nrow, 5, figsize=(16, 12))

c = 0

for i in range(nrow):

    for j in range(5):

        # load data

        seg = pd.read_csv(os.path.join(PATH, "test/" + tests[idx[c]]))

        

        # zscore

        seg = stats.zscore(seg)

        

        # plot

        ax[i, j].plot(seg[::ds], color='g')

        ax[i, j].title.set_text(tests[idx[c]])

        ax[i, j].axis("off")

        

        c += 1
%%time

train_df = pd.read_csv(os.path.join(PATH, "train.csv"), 

                       dtype={"acoustic_data": np.int16, 

                              "time_to_failure": np.float32})
print("Train data has {} rows and {} columns"

      .format(train_df.shape[0], train_df.shape[1]))
# head 10

pd.options.display.precision = 15

train_df.head(10)
# plot all data, downsampled



# fig & ax

fig, ax1 = plt.subplots(figsize=(12, 8))



# acoustic data

ax1.plot(train_df["acoustic_data"].values[::ds], color='g')

ax1.set_ylabel("acoustic data")

ax1.tick_params('y', colors='g')



# time to failure

ax2 = ax1.twinx()

ax2.plot(train_df["time_to_failure"].values[::ds], color='r')

ax2.set_ylabel("time to failure")

ax2.tick_params('y', colors='r')
# too heavy to take them anymore

acousticData = train_df["acoustic_data"][::ds].values

timeToFailure = train_df["time_to_failure"][::ds].values



del train_df
# timing (index) of events

sorted_idx = np.argsort(np.diff(timeToFailure))

sorted_idx = sorted_idx[::-1]

eventTiming = np.sort(sorted_idx[0:15])

print("Timing of events: " + str(eventTiming))
# event triggered average (n = 16)

fig, ax1 = plt.subplots(4, 4, figsize=(12, 8))

c = 0

for i in range(4):

    for j in range(4):

        triggeredRange = range(sorted_idx[c]-int(1500000/ds), sorted_idx[c]+int(500000/ds))



        # acoustic data

        ax1[i, j].plot(acousticData[triggeredRange], color='g')

        ax1[i, j].axis("off")



        # time to failure

        ax2 = ax1[i, j].twinx()

        ax2.plot(timeToFailure[triggeredRange], color='r')

        ax2.axis("off")



        c += 1
# event amplitude (max - min) vs time to failure from peak

amplitude = np.zeros(16)

max_to_failure = np.zeros(16)

min_to_failure = np.zeros(16)

for c in range(16):

    triggeredRange = range(sorted_idx[c]-int(1500000/ds), sorted_idx[c]+int(500000/ds))

    amplitude[c] = np.max(acousticData[triggeredRange]) - np.min(acousticData[triggeredRange])

    max_to_failure[c] = int(1500000/ds) - np.argmax(acousticData[triggeredRange])

    min_to_failure[c] = int(1500000/ds) - np.argmin(acousticData[triggeredRange])

    

# correlation?

fig, ax = plt.subplots(1, 2, figsize=(12, 8))

sns.regplot(x=amplitude, y=max_to_failure, ax=ax[0])

ax[0].set_xlabel("amplitude")

ax[0].set_ylabel("time: event - max")

sns.regplot(x=amplitude, y=min_to_failure, ax=ax[1])

ax[1].set_xlabel("amplitude")

ax[1].set_ylabel("time: event - min")
# spectrogram

fig, ax1 = plt.subplots(4, 4, figsize=(12, 8))

c = 0

for i in range(4):

    for j in range(4):

        triggeredRange = range(sorted_idx[c]-int(1500000/ds), sorted_idx[c])



        # FFT of acoustic data

        f, t, Sxx = spectrogram(acousticData[triggeredRange], ds)

        

        # acoustic data

        ax1[i, j].pcolormesh(t, f, Sxx)

        if j == 0:

            ax1[i, j].set_ylabel("frequency (Hz)")

            ax1[i, j].set_xlabel("")

            ax1[i, j].get_xaxis().set_ticks([])

        else:

            ax1[i, j].axis("off")



        c += 1
# FFT

fig, ax1 = plt.subplots(4, 4, figsize=(12, 8))

c = 0

for i in range(4):

    for j in range(4):

        triggeredRange = range(sorted_idx[c]-int(10000/ds), sorted_idx[c])



        # FFT of acoustic data

        power = 10*np.log10(np.abs(np.fft.rfft(acousticData[triggeredRange])))

        f = np.linspace(0, ds/2, len(power))

        

        # acoustic data

        ax1[i, j].plot(f, power, c='b')

        if j == 0:

            ax1[i, j].set_ylabel("power")

            ax1[i, j].set_xlabel("")

            ax1[i, j].get_xaxis().set_ticks([])

        elif i == 3:

            ax1[i, j].set_ylabel("")

            ax1[i, j].set_xlabel("frequency (Hz)")

            ax1[i, j].get_yaxis().set_ticks([])

        else:

            ax1[i, j].set_ylabel("")

            ax1[i, j].set_xlabel("")

            ax1[i, j].get_xaxis().set_ticks([])

            ax1[i, j].get_yaxis().set_ticks([])



        c += 1