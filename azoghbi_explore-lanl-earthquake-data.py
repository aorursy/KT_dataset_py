import numpy as np 

import pandas as pd

%pylab inline



# Input data files are available in the "../input/" directory.

import os

print(os.listdir("../input"))

# read the training data

# the data is large. Read a small number of rows for now

train_data = pd.read_csv('../input/train.csv', nrows=100000)

train_data.head()


fig, ax = plt.subplots(1,4,figsize=(13,4))

ax[0].plot(train_data.time_to_failure.values)

ax[0].set_xlabel('Index'); ax[0].set_ylabel('time to failure')

ax[1].plot(train_data.acoustic_data.values)

ax[1].set_xlabel('Index'); ax[1].set_ylabel('acoustic data')

ax[2].plot(np.diff(train_data.time_to_failure.values))

ax[2].set_xlabel('Index'); ax[2].set_ylabel('step of time_to_failure')

ax[3].plot(train_data.acoustic_data.values, train_data.time_to_failure.values, 'o', alpha=0.1)

ax[3].set_xlabel('acoustic data'); ax[3].set_ylabel('time to failure')

plt.tight_layout(pad=2)
fig, ax = plt.subplots(1,4,figsize=(13,4))

nplt = 4096*3

ax[0].plot(train_data.time_to_failure.values[:nplt])

ax[0].set_xlabel('Index'); ax[0].set_ylabel('time to failure')

ax[1].plot(train_data.acoustic_data.values[:nplt])

ax[1].set_xlabel('Index'); ax[1].set_ylabel('acoustic data')

ax[2].plot(np.diff(train_data.time_to_failure.values[:nplt]))

ax[2].set_xlabel('Index'); ax[2].set_ylabel('step of time_to_failure')

ax[3].plot(train_data.acoustic_data.values[:nplt], train_data.time_to_failure.values[:nplt], 'o', alpha=0.1)

ax[3].set_xlabel('acoustic data'); ax[3].set_ylabel('time to failure')

plt.tight_layout(pad=2)
# read the training data

# the data is large. Read a small number of rows for now

train_data_long = pd.read_csv('../input/train.csv', nrows=10000000)

train_data_long.head()
# plot every 500 points, so we explore the data quickly

fig, ax = plt.subplots(1,3,figsize=(13,4))

ax[0].plot(train_data_long.time_to_failure.values[::500])

ax[0].set_xlabel('Index'); ax[0].set_ylabel('time to failure')

ax[1].plot(train_data_long.acoustic_data.values[::500])

ax[1].set_xlabel('Index'); ax[1].set_ylabel('acoustic data')

ax[2].plot(train_data_long.acoustic_data.values[::500], train_data_long.time_to_failure.values[::500], 'o', alpha=0.1)

ax[2].set_xlabel('acoustic data'); ax[2].set_ylabel('time to failure')

plt.tight_layout(pad=2)