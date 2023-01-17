# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import pyarrow.parquet as pq
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.
%%time 
# Read in 5 signals
# Each column contains one signal
subset_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(5)]).to_pandas()
subset_train.info()
subset_train.head()
subset_train.tail()
%%time
meta_train = pd.read_csv('../input/metadata_train.csv')
meta_test = pd.read_csv('../input/metadata_test.csv')
meta_train.head()
meta_test.head()
meta_train.isna().sum()
meta_test.isna().sum()
meta_train['signal_id']
meta_test['signal_id']
meta_train['id_measurement'] # ID code for a trio of signals
meta_test['id_measurement']
sns.countplot(meta_train['target']); # Quite unbalanced, 1 => fault in the power line
meta_train['phase'].value_counts() # 3 phase values
meta_test['phase'].value_counts() # 3 phase values
sns.countplot(meta_train['phase']);
sns.countplot(meta_test['phase']);
subset_train.head()
%%time 
subset_test = pq.read_pandas('../input/test.parquet', columns=[str(i) for i in range(8712,8717)]).to_pandas()
subset_test.shape
subset_test.head()
subset_test.tail()
meta_train.head()
# Let's look at some of the signals

fig, ((ax1, ax2), (ax3,ax4)) = plt.subplots(2,2, figsize=(12,12))
ax1.hist(subset_train['0'], bins=100);
ax2.hist(subset_train['1'], bins=100)
ax3.hist(subset_train['2'], bins=100)
ax4.hist(subset_train['3'], bins=100);
# How well does just the phase reflect the target?
meta_train.corr()['target']
# Not very well it seems