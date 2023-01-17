import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split

from tqdm import tqdm

np.random.seed(0)
size = 262144

score_archive = []

diff_archive = []

for i in tqdm(range(10000)):

    # generate synthetic dataset

    rng = np.random.uniform(size=size)

    pred = np.array([i for i in range(size)])

    y = np.array([0 if x<0.975 else 1 for x in rng[:size//2]] + [1 if x<0.975 else 0 for x in rng[size//2:]])

    # make synthetic public set and private set

    pub_pred, pri_pred, pub_y, pri_y = train_test_split(pred, y, shuffle=True, test_size=0.5)

    # calculate public set score

    score_archive.append(roc_auc_score(pub_y, pub_pred))

    # calculate score difference between public set and private set

    diff_archive.append(roc_auc_score(pub_y, pub_pred)-roc_auc_score(pri_y, pri_pred))
plt.figure(figsize=(14,6))

plt.hist(score_archive, bins=100)

plt.title('distribution of roc auc of a perfect classifier')

plt.show()
np.mean(score_archive)
np.percentile(score_archive, np.arange(40, 60, 1))
plt.figure(figsize=(14,6))

plt.hist(diff_archive, bins=100)

plt.title('distribution of roc auc diff between public and private')

plt.show()
np.mean(np.abs(diff_archive))
np.percentile(diff_archive, np.arange(40, 60, 1))