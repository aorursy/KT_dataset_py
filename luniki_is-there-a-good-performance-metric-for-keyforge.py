# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

# print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
metrics = ["winrate", "consistency", "aerc", "ewr", "kfai", "sas"]

df = pd.read_csv("../input/top_200.csv")

df.head(3)
df[metrics].corr("pearson")
#sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1.5)

a4_dims = (11.7, 8.27)

fig, ax = plt.subplots(figsize=a4_dims)



labels = ['Win Rate', 'Consistency', 'AERC', 'EWR', 'KFAI', 'SAS']

g = sns.heatmap(df[metrics].corr("pearson"), square=True, cmap="RdBu_r", annot=True, vmin=0, vmax=1, xticklabels=labels, yticklabels=labels, ax=ax)

plt.yticks(rotation=0)

plt.show()
labels = ['ADHD consistency', 'AERC', 'EWR', 'KFAI', 'SAS']

y_vars = ["consistency", "aerc", "ewr", "kfai", "sas"]

g = sns.PairGrid(df, x_vars=['winrate'], y_vars=y_vars, aspect=1.6, height=5)

g = g.map(plt.scatter, marker="+", vmin=0, vmax=100)

for ax, label in zip(g.axes, labels):

    ax[0].yaxis.set_label_text(label)

    ax[0].xaxis