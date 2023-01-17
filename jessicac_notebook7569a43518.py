

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

import os

import gc # We're gonna be clearing memory a lot

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



df_train = pd.read_csv('../input/clicks_train.csv')

df_test = pd.read_csv('../input/clicks_test.csv')
p = sns.color_palette()

sizes_train = df_train.groupby('display_id')['ad_id'].count().value_counts()

sizes_test = df_test.groupby('display_id')['ad_id'].count().value_counts()

sizes_train = sizes_train / np.sum(sizes_train)

sizes_test = sizes_test / np.sum(sizes_test)



plt.figure(figsize=(12,4))

sns.barplot(sizes_train.index, sizes_train.values, alpha=0.8, color=p[0], label='train')

sns.barplot(sizes_test.index, sizes_test.values, alpha=0.6, color=p[1], label='test')

plt.legend()

plt.xlabel('Number of Ads in display', fontsize=12)

plt.ylabel('Proportion of set', fontsize=12)