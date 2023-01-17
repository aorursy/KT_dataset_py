import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import missingno as msno
from matplotlib_venn import venn2
import seaborn as sns; sns.set()
%matplotlib inline

import os
print(os.listdir("../input"))
df_train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"], engine='c')
df_test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"], engine='c')
print("{} observations and {} features in train set.".format(*df_train.shape))
print("{} observations and {} features in test set.".format(*df_test.shape))
df_train.target.describe()
fig, ax = plt.subplots(figsize=(16, 5))
sns.distplot(df_train.target, ax=ax)
fig, ax = plt.subplots(figsize=(12, 3))
sns.boxplot(x='target', data=df_train)
fig, axs = plt.subplots(ncols=3, figsize=(15, 10))
sns.boxplot(x='feature_1', y='target', data=df_train, orient='v',dodge=False, ax=axs[0])
sns.boxplot(x='feature_2', y='target', data=df_train, orient='v',dodge=False, ax=axs[1])
sns.boxplot(x='feature_3', y='target', data=df_train, orient='v',dodge=False, ax=axs[2])