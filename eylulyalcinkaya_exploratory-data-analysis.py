import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import gc

import seaborn as sns



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
%%time

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
train.describe()
train.rename({"acoustic_data": "signal", "time_to_failure": "time"}, axis="columns", inplace=True)
train_sample_signal = train['signal'].values[::100]

train_sample_time = train['time'].values[::100]



fig, ax1 = plt.subplots(figsize=(20,8))



plt.title("Signal and time to failure with %1 of data")

plt.plot(train_sample_signal, color = 'burlywood')

ax2 = ax1.twinx()

plt.plot(train_sample_time, color = 'g')



del train_sample_signal

del train_sample_time

gc.collect()
train_sample = train.sample(frac=0.01)



plt.figure(figsize=(12,6))

plt.title("Signal data histogram")

ax = sns.distplot(train_sample['signal'], label='Signal')



del train_sample

gc.collect()
train_sample = train.sample(frac=0.01)

plt.figure(figsize=(10,5))

plt.title("Signal distribution without outliers")

tmp = train_sample.signal[train_sample.signal.between(-25, 25)]

ax = sns.distplot(tmp, label='Signal', kde=False)



del train_sample

del tmp

gc.collect()
train_signal_million = train['signal'].values[:1000000]

train_time_million = train['time'].values[:1000000]



fig, ax1 = plt.subplots(figsize=(20, 8))

plt.title("first 1 million rows")

plt.plot(train_signal_million, color = 'burlywood')

ax2 = plt.twinx()

plt.plot(train_time_million, color = 'g')



del train_signal_million

del train_time_million

gc.collect()
train_signal_thousand = train['signal'].values[:100000]

train_time_thousand = train['time'].values[:100000]



fig, ax1 = plt.subplots(figsize=(20, 8))

plt.title("first 100 thousand rows")

plt.plot(train_signal_thousand, color = 'burlywood')

ax2 = plt.twinx()

plt.plot(train_time_thousand, color = 'g')



del train_signal_thousand

del train_time_thousand

gc.collect()
train_signal_one = train['signal'].values[:10000]

train_time_one = train['time'].values[:10000]



fig, ax1 = plt.subplots(figsize=(20, 8))

plt.title("first ten thousand rows")

plt.plot(train_signal_one, color = 'burlywood')

ax2 = plt.twinx()

plt.plot(train_time_one, color = 'g')



del train_signal_one

del train_time_one

gc.collect()
test_files = os.listdir("../input/test")

len(test_files)
seg = pd.read_csv("../input/test/seg_004cd2.csv")



fig, ax1 = plt.subplots(figsize=(20, 8))

plt.title("example test data")

plt.plot(seg, color = 'g')



seg2 = pd.read_csv("../input/test/seg_00c35b.csv")



fig2, ax1 = plt.subplots(figsize=(20, 8))

plt.title("example test data")

plt.plot(seg2, color = 'g')



seg3 = pd.read_csv("../input/test/seg_00cc91.csv")



fig3, ax1 = plt.subplots(figsize=(20, 8))

plt.title("example test data")

plt.plot(seg3, color = 'g')