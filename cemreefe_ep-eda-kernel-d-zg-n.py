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
import matplotlib.pyplot as plt
%%time

train = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
#this one is high def data

df = pd.read_csv("../input/train.csv", nrows=10000000)

train.rename({"acoustic_data": "acd", "time_to_failure": "ttf"}, axis="columns", inplace=True)
acd_small = train['acd'].values[::50]

ttf_small = train['ttf'].values[::50]



fig, ax1 = plt.subplots(figsize=(20, 8))



plt.plot(acd_small, color='pink')

ax2 = ax1.twinx()

plt.plot(ttf_small, color='g')
acd_small = train['acd'].values[::50]

ttf_small = train['ttf'].values[::50]



fig, ax1 = plt.subplots(figsize=(20, 8))



plt.plot(acd_small[:300000], color='pink')

ax2 = ax1.twinx()

plt.plot(ttf_small[:300000], color='g')



smallest_tvals=train[train["ttf"]==0]

print('t: ', (smallest_tvals.shape))



smallest_dvals=df[df["time_to_failure"]==0]

print('d: ', (smallest_dvals.shape))
acd_small = train['acd'].values[::50]

ttf_small = train['ttf'].values[::50]



fig, ax1 = plt.subplots(figsize=(20, 8))



plt.plot(acd_small[:3000], color='pink')

ax2 = ax1.twinx()

plt.plot(ttf_small[:3000], color='g')



acd_small = train['acd'].values[::50]

ttf_small = train['ttf'].values[::50]



fig, ax1 = plt.subplots(figsize=(20, 8))



plt.plot(acd_small[:300], color='pink')

ax2 = ax1.twinx()

plt.plot(ttf_small[:300], color='g')



df.head(4095).plot(y="time_to_failure",title="BEFORE JUMP SEGMENT VALS, 4095 DATA PTS", color = "green")

df.head(4096).tail(2).plot(y="time_to_failure",title="AFTER JUMP SEGMENT VALS, 2 DATA PTS")

print(df.head(1)["time_to_failure"])

print(df.head(4096).tail(3)["time_to_failure"])


