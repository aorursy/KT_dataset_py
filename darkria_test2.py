# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# Get first 10000 rows and print some info about columns
train = pd.read_csv("../input/train.csv", parse_dates=['srch_ci', 'srch_co'], nrows=20000)
train.info()

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")
# distribution of the total number of people per cluster
src_total_cnt = train.srch_adults_cnt + train.srch_children_cnt
train['src_total_cnt'] = src_total_cnt
ax = sns.kdeplot(train['hotel_cluster'], train['src_total_cnt'], cmap="Purples_d")
lim = ax.set(ylim=(0.5, 4.5))
# plot all columns countplots
import numpy as np
rows = train.columns.size//2 - 1
fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12,36))
fig.tight_layout()
i = 0
j = 0
for col in train.columns:
    if j >= 2:
        j = 0
        i += 1
    # avoid to plot by date    
    if train[col].dtype == np.int64:
        sns.countplot(x=col, data=train, ax=axes[i][j])
        j += 1
# putting the two above together
with sns.color_palette("ocean"):
    sns.countplot(x='is_booking', hue='is_mobile', data=train)
# putting the two above together
with sns.color_palette("Set1"):
    sns.countplot(x='cnt', hue='is_booking', data=train)
import numpy as np
# get number of booked nights as difference between check in and check out
is_same = (train['hotel_country'] - train['user_location_country'])*10000000000000000
#is_same = (is_same / np.timedelta64(1, 'D')).astype(float) # convert to float to avoid NA problems
train['is_same'] = is_same
with sns.color_palette("husl"):
    plt.figure(figsize=(11, 9))
    ax = sns.violinplot(x='hotel_continent', y='is_same', data=train)
#lim = ax.set(ylim=(0, 15))
#import numpy as np
# get number of booked nights as difference between check in and check out
plt.figure(figsize=(19, 9))
ax = sns.violinplot(x='hotel_cluster', y='is_same', data=train)
lim = ax.set(xlim=(-1, 50))