# https://github.com/kaggle/docker-python

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

# Input data files are available in the "../input/" directory.

from subprocess import check_output

p = sns.color_palette()
clicks_train = pd.read_csv('../input/clicks_train.csv')

clicks_test = pd.read_csv('../input/clicks_test.csv')
sizes_train = clicks_train.groupby('display_id')['ad_id'].count().value_counts()

sizes_train = sizes_train / np.sum(sizes_train)

plt.figure(figsize=(12,4))

sns.barplot(sizes_train.index, sizes_train.values, alpha=0.8, color=p[0], label='train')

plt.legend()

plt.xlabel('Number of Ads in display', fontsize=12)

plt.ylabel('Proportion of set', fontsize=12);
ad_usage_train = clicks_train.groupby('ad_id')['ad_id'].count()



for i in [2, 10, 50, 100, 1000]:

    print('Ads that appear less than {} times: {}%'.format(i, round((ad_usage_train < i).mean() * 100, 2)))



plt.figure(figsize=(12, 6))

plt.hist(ad_usage_train.values, bins=50, log=True)

plt.xlabel('Number of times ad appeared', fontsize=12)

plt.ylabel('log(Count of displays with ad)', fontsize=12)

plt.show();
print('display ids in train:', len(clicks_train.display_id.unique()))

print('display ids in test:', len(clicks_test.display_id.unique()))

print('ad ids in train:', len(clicks_train.ad_id.unique()))

print('ad ids in test:', len(clicks_test.ad_id.unique()))
ad_ctr = clicks_train.groupby('ad_id').clicked.mean().to_frame()

clicks_test_ctr = clicks_test.join(ad_ctr, on='ad_id')

clicks_test_ctr.clicked.fillna(0., inplace=True)
clicks_test_ctr.sort_values(['display_id','clicked'], inplace=True, ascending=False)

subm = clicks_test_ctr.groupby('display_id').ad_id.apply(lambda x: " ".join(map(str,x))).reset_index()

subm.to_csv("subm.csv", index=False)
check_output(['ls','-lh'])


