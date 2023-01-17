# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/UNSW_NB15_training-set.csv")

print(df.info())
print(df.describe())
print(df.head(5))
print(df.proto.value_counts())
f, ax = plt.subplots(1, 1, figsize=(8, 8))

df.proto.value_counts().plot(kind='bar', title="protocol type", ax=ax, figsize=(8,8))

plt.show()
ax = df.groupby(['proto', 'attack_cat']).size().plot(kind='bar')

ax.set_title("# of attack_cat per protocol")

ax.set_xlabel("(proto, attack_cat)")

ax.set_ylabel("Count of attack-cat")

for p in ax.patches:

    ax.annotate(str(format(int(p.get_height()), ',d')), (p.get_x(), p.get_height()*1.01))
# only  protocol type ('tcp', 'udp')

tmp = df.loc[(df['proto'].isin(['tcp', 'udp'])),:]

# tmp.drop(['step', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

# tmp = tmp.reset_index(drop=True)

print(tmp.head(3))