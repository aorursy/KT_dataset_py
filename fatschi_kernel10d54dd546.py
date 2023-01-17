# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn')



df = pd.read_csv("../input/fullcsv/2013-11-03_tromso_stromsgodset_raw_first.csv", sep=',', names=["timestamp", "tag_id", "x", "y", "heading","direction","energy","speed","total_distance"])



df_sample = df.sample(10000)



sns.kdeplot(df_sample.x, df_sample.y, cmap="Reds", shade=True)
g = sns.FacetGrid(df, col="tag_id", hue="tag_id")

g = (g.map(plt.scatter, "x", "y", edgecolor="w"))



df.groupby('tag_id')['speed'].agg(['mean','median']).plot(kind='bar')