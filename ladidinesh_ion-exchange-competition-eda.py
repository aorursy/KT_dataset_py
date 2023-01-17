import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="whitegrid")
base_dir = "/kaggle/input/liverpool-ion-switching"

print(os.listdir(base_dir))
train = pd.read_csv(f"{base_dir}/train.csv")

test = pd.read_csv(f"{base_dir}/test.csv")

sample_submission = pd.read_csv(f"{base_dir}/sample_submission.csv")

print("Train Dimensions: ",train.shape)

print("Test Dimensions: ",test.shape)
train.head(10)
channels_dist = train['open_channels'].value_counts().rename_axis('Channel').reset_index(name='count')



plt.figure(figsize=(12, 6))

sns.barplot(x = 'Channel', y = 'count', data = channels_dist,palette="Blues_d")

plt.title("Count of open channels in train data")

plt.show()
fig,ax = plt.subplots(ncols=1, nrows=2,figsize=(16,10))

sns.lineplot(x="time", y="signal", data=train[train['time'] <=10], ax = ax[0])

sns.lineplot(x="time", y="open_channels", data=train[train['time'] <=10], ax = ax[1])

plt.show()
train[train['time'] <=50]['open_channels'].value_counts()
train['batch'] =  pd.cut(train['time'],10, labels = list(range(1,11)))
grid_data  = train.groupby(['batch','open_channels']).count().reset_index()

grid_data = grid_data.rename(columns = {'time':'count'})

plt.figure(figsize = (16,16))

g = sns.FacetGrid(grid_data, col="batch", col_wrap=3, height=5)

g = g.map(plt.bar, "open_channels", "count")

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(20,10))

ax1.plot(train["time"], train["signal"], color="blue")

ax1.set_title('Signal',fontsize=20)

ax2.plot(train["time"], train["open_channels"], color="blue")

ax2.set_title('Open Channels', fontsize=20)

plt.xlabel("Time", fontsize=20)

plt.show()