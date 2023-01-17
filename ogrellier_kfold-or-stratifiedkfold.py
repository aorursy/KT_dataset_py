import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

%matplotlib inline
trn = pd.read_csv("../input/train.csv")

target = trn.target.copy()

target.sort_values(inplace=True)
# Create folds

folds = KFold(n_splits=3, shuffle=True, random_state = 45)

# Go through folds

plt.figure(figsize=(10,10))

for trn_idx, val_idx in folds.split(target, target):

    # Stack training target and validation target and plot them

    plt.plot(np.hstack((target.iloc[trn_idx].values, target.iloc[val_idx].values)))

plt.title("KFold Shuffle=True ?")
plt.figure(figsize=(15,7))

from matplotlib.gridspec import GridSpec

gs = GridSpec(1, 2)

ax1 = plt.subplot(gs[0, 0])

# Create folds

kfolds = KFold(n_splits=2, shuffle=False, random_state=2645312378)

# Go through folds

idx = np.arange(len(target))

for trn_idx, val_idx in kfolds.split(trn.values):

    # Stack training target and validation target and plot them

    ax1.plot(np.hstack((idx[trn_idx], idx[val_idx])))

ax1.set_title("KFold Shuffle=False")



ax2 = plt.subplot(gs[0, 1])

# Create folds

kfolds = KFold(n_splits=2, shuffle=True, random_state=2645312378)

# Go through folds

idx = np.arange(len(target))

for I, (trn_idx, val_idx) in enumerate(kfolds.split(trn.values)):

    # Stack training target and validation target and plot them

    ax2.plot(np.hstack((idx[trn_idx], idx[val_idx])) + 20000 * I)

ax2.set_title("KFold Shuffle=False")
# Create folds

plt.figure(figsize=(10,10))

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state = 5)

# Go through folds

for trn_idx, val_idx in folds.split(target, target):

    # Stack training target and validation target and plot them

    plt.plot(np.hstack((target.iloc[trn_idx].values, target.iloc[val_idx].values)))

plt.title("StratifiedKFold Shuffle=True ?")
idx = target.index.values

np.random.shuffle(idx)

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state = 5)

# Go through folds

plt.figure(figsize=(10,10))

for trn_idx, val_idx in folds.split(target, target):

    # Stack training target and validation target and plot them

    plt.plot(np.hstack((target.loc[idx[trn_idx]].values, 

                        target.loc[idx[val_idx]].values)))

plt.title("StratifiedKFold Shuffle=True ?")