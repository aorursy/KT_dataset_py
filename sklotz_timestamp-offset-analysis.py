# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

fb_train = pd.read_csv('../input/train.csv')
fb_test = pd.read_csv('../input/test.csv')

# Any results you write to the current directory are saved as output.

fb_train.describe()
common_places = fb_train.place_id.value_counts()[0:30].index
train_common = fb_train[fb_train.place_id.isin(common_places)]
train_common.shape
train_common.describe()
day_offset = 9*60
week_offset = 60*24*3
complete_offset = day_offset + week_offset

number_of_places_to_display = 12
plt.figure(figsize=(18,21))
plt.title('Daily check-ins')
for i in range(0,number_of_places_to_display):
    ax = plt.subplot(number_of_places_to_display/3,3,i+1)
    ax.set_xlim([0, 60*24])
    sns.distplot((train_common[train_common.place_id==common_places[i]].time + complete_offset)%(60*24), bins=100)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.xticks(np.arange(0, 60*24, 60))
    labels = [i for i in range(0,25)]
    ax = plt.gca()
    ax.set_xticklabels(labels)
    plt.title("pid: " + str(common_places[i]))
plt.figure(figsize=(18,21))
plt.title('Weekly check-ins')
for i in range(0,number_of_places_to_display):
    ax = plt.subplot(number_of_places_to_display/3,3,i+1)
    ax.set_xlim([0, 60*24*7])
    sns.distplot((train_common[train_common.place_id==common_places[i]].time + complete_offset)%(60*24*7), bins=100)
    plt.xlabel("Time")
    plt.ylabel("Fraction")
    plt.gca().get_xaxis().set_ticks([])
    plt.title("pid: " + str(common_places[i]))
plt.figure(figsize=(18,9))

plt.subplot(211)
sns.distplot((train_common.time + complete_offset)%(60*24), bins=100)
plt.xticks(np.arange(0, 60*24, 60))
labels = [i for i in range(0,25)]
ax = plt.gca()
ax.set_xticklabels(labels)

plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Daily check-ins of all common places")

plt.subplot(212)
sns.distplot((train_common.time + complete_offset)%(60*24*7), bins=100)
plt.xticks(np.arange(0, 60*24*7, int(60*24/4)))
labels = [i*6%24 for i in range(0,int(7*24/4+1))]
ax = plt.gca()
ax.set_xticklabels(labels)

plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Weekly check-ins of all common places")
plt.figure(figsize=(18,9))

plt.subplot(211)
sns.distplot((train_common.time + complete_offset)%(60*24), bins=24)
plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Daily check-ins of all common places")
plt.xticks(np.arange(0, 60*24, 60))
labels = [i for i in range(0,25)]
ax = plt.gca()
ax.set_xticklabels(labels)

plt.subplot(212)
sns.distplot((train_common.time + complete_offset)%(60*24*7), bins=7)
plt.xticks(np.arange(0, 60*24*7, 60*24))
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = plt.gca()
ax.set_xticklabels(labels)
plt.xlabel("Weekday")
plt.ylabel("Fraction")
plt.title("Weekly check-ins of all common places")
fb_train['day'] = (fb_train.time + complete_offset)  % (60*24*7) / (60*24)
fb_train['day'] = fb_train['day'].apply(np.floor)

fb_train['hour'] = (fb_train.time + complete_offset)  % (60*24) / 60
fb_train['hour'] = fb_train['hour'].apply(np.floor)
fb_train.describe()
plt.figure(figsize=(18,4))
ax = plt.subplot()
ax.set_xlim([-1, 7])
sns.distplot(fb_train.day, kde=False)
plt.xticks(np.arange(0, 7, 1))
labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
ax = plt.gca()
ax.set_xticklabels(labels)
plt.xlabel("Weekday")
plt.ylabel("Fraction")
plt.title("Weekly check-ins of all places")
plt.figure(figsize=(18,4))
ax = plt.subplot()
ax.set_xlim([-1, 24])
sns.distplot(fb_train.hour, kde=False)
plt.xlabel("Time in h")
plt.ylabel("Fraction")
plt.title("Daily check-ins of all places")
plt.xticks(np.arange(0, 25, 1))
labels = [i for i in range(0,25)]
ax = plt.gca()
ax.set_xticklabels(labels)