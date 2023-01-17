# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import time

import datetime



import numpy as np # linear algebra

from scipy import stats

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]



train_df["pickup_datetime"] = pd.to_datetime(train_df["pickup_datetime"])

train_df["year"] = train_df["pickup_datetime"].dt.year

train_df["month"] = train_df["pickup_datetime"].dt.month

train_df["day"] = train_df["pickup_datetime"].dt.day

train_df["hour"] = train_df["pickup_datetime"].dt.hour

train_df["weekday"] = train_df["pickup_datetime"].dt.weekday



train_df["daytime"] = 0

train_df.loc[(train_df.hour >= 0) & (train_df.hour < 4), 'daytime'] = 0

train_df.loc[(train_df.hour >= 4) & (train_df.hour < 8), 'daytime'] = 1

train_df.loc[(train_df.hour >= 8) & (train_df.hour < 12), 'daytime'] = 2

train_df.loc[(train_df.hour >= 12) & (train_df.hour < 16), 'daytime'] = 3

train_df.loc[(train_df.hour >= 16) & (train_df.hour < 20), 'daytime'] = 4

train_df.loc[(train_df.hour >= 20) & (train_df.hour < 24), 'daytime'] = 5



mini_data = train_df[:50000]

#print(mini_data)

fig, ax = plt.subplots(2, 1) 

ax[0].scatter(mini_data['hour'],mini_data['trip_duration'], alpha=0.1)

ax[0].set_xlabel('hour')

ax[0].set_ylabel('trip_duration')

ax[0].set_ylim([0, 8000])



ax[1].scatter(mini_data['weekday'],mini_data['trip_duration'], alpha=0.1)

ax[1].set_xlabel('weekday')

ax[1].set_ylabel('trip_duration')

ax[1].set_ylim([0, 8000])



plt.show()

train_df.describe()



train_wd0 = train_df[train_df["weekday"] == 0]

train_wd1 = train_df[train_df["weekday"] == 1]

train_wd2 = train_df[train_df["weekday"] == 2]

train_wd3 = train_df[train_df["weekday"] == 3]

train_wd4 = train_df[train_df["weekday"] == 4]

train_wd5 = train_df[train_df["weekday"] == 5]

train_wd6 = train_df[train_df["weekday"] == 6]



for i in range(0,23):

    print((train_df[train_df["hour"] == i]).mean()['trip_duration'])



#g = sns.FacetGrid(mini_data, col='hour')

#g.map(plt.hist, 'trip_duration', bins=200)

#g.set(xlim=(0, 1800))