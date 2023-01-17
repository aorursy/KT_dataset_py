import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from random import shuffle

import matplotlib.pyplot as plt

import numba

import seaborn as sns



# References: 

# https://www.kaggle.com/jsaguiar/seismic-data-exploration 

# https://www.kaggle.com/eylulyalcinkaya/exploratory-data-analysis

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



print(os.listdir("../input"))
test_folder_files = os.listdir("../input/test")



print("\nNumber of files in the test folder", len(test_folder_files))
train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

print("train shape", train.shape)

pd.set_option("display.precision", 15)  # show more decimals

train.head()
train.acoustic_data.describe()

@numba.jit

def get_stats(arr):

    """Memory efficient stats (min, max and mean). """

    size  = len(arr)

    min_value = max_value = arr[0]

    mean_value = 0

    for i in numba.prange(size):

        if arr[i] < min_value:

            min_value = arr[i]

        if arr[i] > max_value:

            max_value = arr[i]

        mean_value += arr[i]

    return min_value, max_value, mean_value/size
tmin, tmax, tmean = get_stats(train.acoustic_data.values)

print("min value: {:.6f}, max value: {:.2f}, mean: {:.4f}".format(tmin, tmax, tmean))
tmin, tmax, tmean = get_stats(train.time_to_failure.values)

print("min value: {:.6f}, max value: {:.2f}, mean: {:.4f}".format(tmin, tmax, tmean))
def single_timeseries(final_idx, init_idx=0, step=1, title="",

                      color1='orange', color2='g'):

    idx = [i for i in range(init_idx, final_idx, step)]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    fig.suptitle(title, fontsize=14)

    

    ax2 = ax1.twinx()

    ax1.set_xlabel('index')

    ax1.set_ylabel('Acoustic data')

    ax2.set_ylabel('Time to failure')

    p1 = sns.lineplot(data=train.iloc[idx].acoustic_data.values, ax=ax1, color=color1)

    p2 = sns.lineplot(data=train.iloc[idx].time_to_failure.values, ax=ax2, color=color2)
single_timeseries(629145000, step=1000, title="Signal and time to failure with all training data")