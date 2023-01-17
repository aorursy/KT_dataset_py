import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

train = pd.read_csv("/kaggle/input/lish-moa/train_features.csv")

train_target = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")

test = pd.read_csv("../input/lish-moa/test_features.csv")

sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

train
train_target
test
sub
train.isnull().values.any(), test.isnull().values.any(), train_target.isnull().values.any()
train["sig_id"].nunique() == len(train)
all(train["sig_id"] == train_target["sig_id"])
all(test["sig_id"] == sub["sig_id"])
train
gcols = [gc for gc in train.columns if "g-" in gc]

ccols = [cc for cc in train.columns if "c-" in cc]

cpcols = [cp for cp in train.columns if "cp_" in cp]
train.describe()
train.info()
# Get columns' names

gcols = [g for g in train.columns if "g-" in g]

ccols = [c for c in train.columns if "c-" in c]

cpcols = [cp for cp in train.columns if "cp_" in cp]

print(len(gcols), len(ccols), len(cpcols))
train.nunique(dropna=False).sort_values()
# # This code is used to check duplicate columns (if any). It runs for a long time: the result is None, so avoid running this cell



# train_factorized = pd.DataFrame(index=train.index)

# for col in tqdm.notebook.tqdm(train.columns):

#     train_factorized[col] = train[col].map(train[col].value_counts())



# dup_cols = {}



# for i, c1 in enumerate(tqdm_notebook(train_factorized.columns)):

#     for c2 in train_factorized.columns[i + 1:]:

#         if c2 not in dup_cols and np.all(train_factorized[c1] == train_factorized[c2]):

#             dup_cols[c2] = c1

            

# dup_cols
%%capture

!pip install seaborn --upgrade
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("darkgrid")



f, axes = plt.subplots(1, 3, figsize=(20, 7))

for ax, col in zip(axes, cpcols):

    sns.countplot(x = col, data=train, ax = ax)
f, axes = plt.subplots(1, 4, figsize=(20, 7))

for idx, ax in enumerate(axes):

    sns.histplot(train["g-" + str(idx)], ax = ax, stat="density", kde=True)

    sns.histplot(test["g-" + str(idx)], ax = ax, color="red", stat="density", kde=True)
f, axes = plt.subplots(1, 4, figsize=(20, 7))

for idx, ax in enumerate(axes):

    sns.histplot(train["c-" + str(idx)], ax = ax, stat="density", kde=True)

    sns.histplot(test["c-" + str(idx)], ax = ax, color="red", stat="density", kde=True)
from scipy.stats import ks_2samp



gcols_diff = []

gcols_kstest = []

for gcol in gcols:

    gcols_kstest.append(ks_2samp(train[gcol], test[gcol])[0])

    if ks_2samp(train[gcol], test[gcol])[0] > 0.03:

        gcols_diff.append(gcol)
from scipy.stats import describe



np.percentile(gcols_kstest, 90)
f, axes = plt.subplots(4, figsize=(12, 20))

for x, ax in enumerate(axes):

    sns.histplot(train[gcols_diff[x]], ax = ax, stat="density", kde=True)

    sns.histplot(test[gcols_diff[x]], ax = ax, color="red", stat="density", kde=True)
f, axes = plt.subplots(2, 1, figsize=(20, 15))

sns.countplot(x="cp_type", data=train, hue="cp_time", ax = axes[0])

sns.countplot(x="cp_dose", data=train, hue="cp_time", ax = axes[1])
train_tar = train_target.iloc[:, 1:]

train_tar
cnt = train_tar[train_tar == 1].sum().sort_values()

cnt
ctlvehicle_idx = train["cp_type"] == "ctl_vehicle"

all(train_target.iloc[:, 1:].sum(axis=1).loc[ctlvehicle_idx] == 0)
train_rowsumcnt = (train_target.iloc[:, 1:].sum(axis=1).value_counts() / len(train) * 100).reset_index().rename(columns={"index": "count", 0: "percentage"})

plt.figure(figsize=(10,7))

sns.barplot(x = "count" , y = "percentage", data=train_rowsumcnt)
train_rowsumcnt
sum(cnt) / train_tar.size
train_colsumcnt = train_target.iloc[:, 1:].sum(axis=0).sort_values().reset_index(name="count")

f, axes = plt.subplots(1, 2, figsize=(22, 10))



chart1 = sns.barplot(x="index", y = "count", data=train_colsumcnt.head(), ax = axes[0])

chart1.set_xticklabels(chart1.get_xticklabels(), rotation=-45, ha="center")

chart2 = sns.barplot(x="index", y = "count", data=train_colsumcnt.tail(), ax = axes[1])

chart2.set_xticklabels(chart2.get_xticklabels(), rotation=-45, ha="center")