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
df_train = pd.read_csv("../input/train.csv")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(15,15))

df_train[df_train['feature_2'] == 1].groupby('feature_2').target.hist(ax=axes[0])
df_train[df_train['feature_2'] == 2].groupby('feature_2').target.hist(ax=axes[1])
df_train[df_train['feature_2'] == 3].groupby('feature_2').target.hist(ax=axes[2])

















df_train['feature_2'].value_counts().head(10).plot.bar()

df_train['feature_3'].value_counts().head(10).plot.bar()





