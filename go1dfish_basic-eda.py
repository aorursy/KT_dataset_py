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
%matplotlib inline
import seaborn as sns
import pyarrow.parquet as pq
import gc
train_meta_df = pd.read_csv("../input/metadata_train.csv")
test_meta_df = pd.read_csv("../input/metadata_test.csv")
print("metadata_train shape is {}".format(train_meta_df.shape))
print("metadata_test shape is {}".format(test_meta_df.shape))
train_meta_df.head(6)
test_meta_df.head()
train_meta_df.isnull().sum()
test_meta_df.isnull().sum()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
sns.countplot(x="target", data=train_meta_df, ax=ax1)
sns.countplot(x="target", data=train_meta_df, hue="phase", ax=ax2)
target_count = train_meta_df.target.value_counts()
print("negative(target=0) target: {}".format(target_count[0]))
print("positive(target=1) target: {}".format(target_count[1]))
print("positive data {:.3}".format((target_count[1]/(target_count[0]+target_count[1]))*100))
miss = train_meta_df.groupby(["id_measurement"]).sum().query("target != 3 & target != 0")
print("not all postive or negative num: {}".format(miss.shape[0]))
miss
print("id_measurement have {} uniques in train".format(train_meta_df.id_measurement.nunique()))
print("id_measurement have {} uniques in test".format(test_meta_df.id_measurement.nunique()))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
g = sns.catplot(x="id_measurement", data=train_meta_df, ax=ax1, kind="count")
label = list(range(train_meta_df.id_measurement.min(), train_meta_df.id_measurement.max(), 1000))
ax1.set_xticks(label, [str(i) for i in label])
ax1.patch.set_facecolor('green')
ax1.patch.set_alpha(0.2)
plt.close(g.fig)
g = sns.catplot(x="id_measurement", data=test_meta_df, ax=ax2, kind="count")
label = list(range(test_meta_df.id_measurement.min(), test_meta_df.id_measurement.max(), 1000))
ax2.set_xticks(label, [str(i) for i in label])
ax2.patch.set_facecolor('yellow')
ax2.patch.set_alpha(0.2)
plt.close(g.fig)
train_meta_df.id_measurement.value_counts().describe()
test_meta_df.id_measurement.value_counts().describe()
print("phase have {} uniques in train".format(train_meta_df.phase.unique()))
print("phase have {} uniques in test".format(test_meta_df.phase.unique()))
print("they are phase numbering")
gc.collect()
subset_train_df = pq.read_pandas('../input/train.parquet').to_pandas()
nan = 0
for col in range(len(subset_train_df.columns)):
    nan += np.count_nonzero(subset_train_df.loc[col, :].isnull())
print("train.parquet have {} nulls".format(nan))
print("train.parquet shape is {}".format(subset_train_df.shape))
subset_train_df.head()
subset_train_df = subset_train_df.T
print("train shape is {}".format(subset_train_df.shape))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)
for i in range(3):
    sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[i, :], ax=ax1, label=["phase:"+str(train_meta_df.iloc[i, :].phase)])
ax1.set_xlabel("example of undamaged signal", fontsize=18)
ax1.set_ylabel("amp", fontsize=18)
ax1.patch.set_facecolor('blue')
ax1.patch.set_alpha(0.2)
for i in range(3, 6):
    sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[i, :], ax=ax2, label=["phase:"+str(train_meta_df.iloc[i, :].phase)])
ax2.set_xlabel("example of damaged signal", fontsize=18)
ax2.set_ylabel("amp", fontsize=18)
ax2.patch.set_facecolor('red')
ax2.patch.set_alpha(0.2)
neg_index = train_meta_df.query("target == 0 & phase == 0").head(9).index.values
pos_index = train_meta_df.query("target == 1 & phase == 0").head(9).index.values
fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
fig.suptitle("Undamaged examples", size=18)
for x, index in enumerate(neg_index):
    for phase in range(3):
        sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[index+phase, :], ax=axes[x//3, x%3])
    axes[x//3, x%3].patch.set_facecolor('blue')
    axes[x//3, x%3].patch.set_alpha(0.2)
fig, axes = plt.subplots(3, 3, figsize=(20, 12), sharex=True, sharey=True)
fig.suptitle("Damaged examples", size=18)
for x, index in enumerate(pos_index):
    for phase in range(3):
        sns.lineplot(x=subset_train_df.columns, y=subset_train_df.iloc[index+phase, :], ax=axes[x//3, x%3])
        axes[x//3, x%3].patch.set_facecolor('red')
        axes[x//3, x%3].patch.set_alpha(0.2)
del subset_train_df, fig, axes
gc.collect()
INPUT_NUM = 3390
TRAIN_NUM = 8712
shapes = []
nulls = 0
subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("1st of the six test.parquet shape is {}".format(subset_test_df.shape))
print("1st of the six test.parquet have {} nulls".format(nan))
del subset_test_df
gc.collect()
subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM, TRAIN_NUM + INPUT_NUM*2)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("2nd of the six test.parquet shape is {}".format(subset_test_df.shape))
print("2nd of the six test.parquet have {} nulls".format(nan))

del subset_test_df
gc.collect()
subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*2, TRAIN_NUM + INPUT_NUM*3)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("3rd of the six test.parquet shape is {}".format(subset_test_df.shape))
print("3rd of the six test.parquet have {} nulls".format(nan))

del subset_test_df
gc.collect()
subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*3, TRAIN_NUM + INPUT_NUM*4)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("4th of the six test.parquet shape is {}".format(subset_test_df.shape))
print("4th of the six test.parquet have {} nulls".format(nan))

del subset_test_df
gc.collect()
subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*4, TRAIN_NUM + INPUT_NUM*5)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("5th of the six test.parquet shape is {}".format(subset_test_df.shape))
print("5th of the six test.parquet have {} nulls".format(nan))

del subset_test_df
gc.collect()
subset_test_df = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(TRAIN_NUM + INPUT_NUM*5, TRAIN_NUM + 20337)]).to_pandas()
nan = 0
for col in range(len(subset_test_df.columns)):
    nan += np.count_nonzero(subset_test_df.loc[col, :].isnull())
shapes.append(subset_test_df.shape)
nulls += nan
print("6th of the six test.parquet shape is {}".format(subset_test_df.shape))
print("6th of the six test.parquet have {} nulls".format(nan))
print("train.parquet have {} nulls".format(nulls))
index = 0
for shape in shapes:
    index += shape[1]
print("train.parquet shape is ({}, {})".format(index, shapes[0][0]))
print("test")
