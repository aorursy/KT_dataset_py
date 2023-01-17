# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import matplotlib.pyplot as plt

import pandas as pandas

import seaborn as sns
train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_targets_scored = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

train_targets_nonscored = pd.read_csv("/kaggle/input/lish-moa/train_targets_nonscored.csv")

train.info()
train.head()
d1 = train.groupby(['cp_type']).size()

d2 = train.groupby(['cp_time']).size()

d3 = train.groupby(['cp_dose']).size()



fig, ax = plt.subplots(1, 3, figsize=(12,4))

sns.barplot(x = d1.index, y = d1, ax = ax[0])

sns.barplot(x = d2.index, y = d2, ax = ax[1])

sns.barplot(x = d3.index, y = d3, ax = ax[2])
train.info()
train_targets_scored.head()
"""

The number of rows in the train_features.csv, and train_targets_scored.csv are the same, and 

they are ordered in the same way.

"""

(train_targets_scored.sig_id == train.sig_id).sum() 
"""

Each column in train_targets_scored.csv represents a feature that may be present in the compound. As can be 

seen below, there can be more than one feature present in the compound.

"""

train[train.columns[1:]].sum(axis=1).head()
train_targets_nonscored.head()
train = train.merge(train_targets_scored, left_on='sig_id', right_on='sig_id', how='inner')
train.index = train.sig_id
target_names = train_targets_scored.columns[1:]
d1 = train[train['cp_type'] == 'ctl_vehicle']

d1 = d1[target_names]

d1 = d1.sum(axis=1)

d1 = sum(d1)

print("The number of targets present in the control samples: " + str(d1))
d1 = train.groupby(['cp_time']).mean()

d1 = d1[target_names].mean(axis=1)

sns.barplot(x=d1.index, y=d1)
d1 = train.groupby(['cp_dose']).mean()

d1 = d1[target_names].mean(axis=1)

sns.barplot(x=d1.index, y=d1)