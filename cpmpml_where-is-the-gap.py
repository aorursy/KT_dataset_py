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
from matplotlib import pyplot as plt
%matplotlib inline

import seaborn as sns
train = pd.read_csv('../input/training_set.csv')
train_meta = pd.read_csv('../input/training_set_metadata.csv')
train = train.merge(train_meta[['object_id', 'ddf', 'ra', 'decl', 'target']], 
                    how='left', on='object_id')
object_id = 105869076
print('object_id', object_id)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
df = train[train.object_id == object_id]
ax.scatter(df.mjd, df.flux, c=df.passband, cmap='rainbow', marker='+')
plt.show()
train['mjd_frac'] = np.modf(train.mjd)[0]

_ = plt.hist(train.mjd_frac, bins=100)

train['mjd_frac'] = np.modf(train.mjd + 0.3)[0]

_ = plt.hist(train.mjd_frac, bins=100)
train['mjd_night'] = np.modf(train.mjd + 0.3)[1]
def middle_gap(s):
    s = s.values
    s_prev = np.roll(s, 1)
    s_delta = s - s_prev
    s_delta_max = np.argmax(s_delta)
    s_middle_gap = (s[s_delta_max] + s_prev[s_delta_max]) / 2
    s_middle_gap = np.modf(s_middle_gap / 365)[0] * 365
    return s_middle_gap
    
train['middle_gap'] = train.groupby('object_id'
                                   ).mjd_night.transform(middle_gap)
fix, ax = plt.subplots(1, 1, figsize=(12, 12))
plt.title('Middle gap')
plt.ylabel('Decl')
plt.xlabel('Ra')
ax.scatter(train.ra, train.decl, 
           cmap='rainbow', c=train.middle_gap, marker='+')
fix, ax = plt.subplots(1, 1, figsize=(12, 12))
plt.ylabel('Middle gap')
plt.xlabel('Ra')
ax.scatter(train.ra, train.middle_gap, marker='+')