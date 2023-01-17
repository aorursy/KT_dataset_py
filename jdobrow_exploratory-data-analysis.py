import pandas as pd

import numpy as np



from scipy import stats

import math

import random



import matplotlib.pyplot as plt

%matplotlib inline
train_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

test_df = pd.read_csv('/kaggle/input/liverpool-ion-switching/test.csv')
train_df.head()
train_df.tail()
train_df.info(null_counts=True)
test_df.info(null_counts=True)
train_df.describe()
test_df.describe()
plt.hist(train_df.time)

plt.hist(test_df.time)

plt.legend(labels=['Train', 'Test'])

plt.title('Time Distribution (Just Checking)')

plt.show()
train_df['open_channels'].value_counts().plot(kind='bar')

plt.title('Open channels distribution')

plt.show()
scores = []

for iteration in range(1000000):    

    total = 0

    for choice in range(10):

        if random.randint(0,10) > 9:

            total += 1

    scores.append(total)

plt.hist(scores, bins=22)

plt.title('Binomial Distribution')

plt.show()
plt.figure(figsize=(6,6))

plt.hist(train_df.signal, bins=20)

plt.hist(test_df.signal, bins=20)

plt.title('Signal Distribution for Test and Train')

plt.legend(labels=['Train', 'Test'])

plt.show()

print('Train mean {}, median {}, standard deviation {}'.format(np.mean(train_df.signal), np.median(train_df.signal), np.std(train_df.signal)))

print('Test mean {}, median {}, standard deviation {}'.format(np.mean(test_df.signal), np.median(test_df.signal), np.std(test_df.signal)))

print('\nTrain:', stats.normaltest(train_df.signal))

print('Test:', stats.normaltest(train_df.signal))
stats.ttest_ind(train_df.signal, test_df.signal)
plt.figure(figsize=(18,18))

plt.plot(train_df.signal[train_df.time < 2])

plt.plot(train_df.open_channels[train_df.time < 2])

plt.show()
start = 0.72

end = 0.727

plt.figure(figsize=(14,14))

plt.plot(train_df.signal[(train_df.time > start) & (train_df.time < end)])

plt.plot(train_df.open_channels[(train_df.time > start) & (train_df.time < end)])

plt.legend(['Signal Strength', 'Number of Open Channels'])

plt.show()
start = 200.07

end = 200.08

plt.figure(figsize=(14,14))

plt.plot(train_df.signal[(train_df.time > start) & (train_df.time < end)])

plt.plot(train_df.open_channels[(train_df.time > start) & (train_df.time < end)])

plt.legend(['Signal Strength', 'Number of Open Channels'])

plt.show()
start = 310.07

end = 310.08

plt.figure(figsize=(14,14))

plt.plot(train_df.signal[(train_df.time > start) & (train_df.time < end)])

plt.plot(train_df.open_channels[(train_df.time > start) & (train_df.time < end)])

plt.legend(['Signal Strength', 'Number of Open Channels'])

plt.show()
train_df['batch'] = 0

for i in range(0, 10):

    train_df.iloc[i * 500000: 500000 * (i + 1), 3] = i

    

test_df['batch'] = 0

for i in range(0, 4):

    test_df.iloc[i * 500000: 500000 * (i + 1), 2] = i
fig, axes = plt.subplots(2, 5, figsize=(16, 8))

num_batches = len(train_df.batch.unique())

fig.suptitle('Signal and Open Channels by Batch. Blue == signal, Orange == open_channels', fontsize=16)

axis_on = True

for i in range(num_batches):

    axes[i // (num_batches // 2), i % (num_batches // 2)].plot(train_df.signal[train_df.batch == i])

    axes[i // (num_batches // 2), i % (num_batches // 2)].plot(train_df.open_channels[train_df.batch == i])

    axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks(range(-4, 13))

    if axis_on == False:

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_xticks([])

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks([])

    axis_on = False
test_df['batch'] = 0

for i in range(0, 4):

    test_df.iloc[i * 500000: 500000 * (i + 1), 2] = i



fig, axes = plt.subplots(2, 2, figsize=(10, 10))

num_batches = len(test_df.batch.unique())

fig.suptitle('Test Distributions', fontsize=16)

axis_on = True

for i in range(num_batches):

    axes[i // (num_batches // 2), i % (num_batches // 2)].plot(test_df.signal[test_df.batch == i])

    axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks(range(-4, 13))

    if axis_on == False:

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_xticks([])

        axes[i // (num_batches // 2), i % (num_batches // 2)].set_yticks([])

    axis_on = False
fig, axes = plt.subplots(4, 3, figsize=(15, 20))

fig.suptitle('Signal Distributions at Number of Open Channels', fontsize=16)

for i in range(11):

    n, bins, patches = axes[i // 3, i % 3].hist(train_df.signal[train_df.open_channels == i], bins=40)

    ind = list(n).index(max(n))

    mean = round(np.mean(train_df.signal[train_df.open_channels == i]), 2)

    binned_mode = (bins[ind] + bins[ind + 1])/2

    axes[i // 3, i % 3].set_title('Channels {}, BinMode {}, Mean {}'.format(i, round(binned_mode, 2), mean))

    axes[i // 3, i % 3].set_xticks([-5, -2.5, 0, 2.5, 5, 7.5, 10, 12.5])

    axes[i // 3, i % 3].axvline(binned_mode , color='orange')

    axes[i // 3, i % 3].axvline(mean , color='green')

plt.show()