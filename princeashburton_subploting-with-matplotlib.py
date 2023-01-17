# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(20)
reviews.info()
fig, axarr = plt.subplots(2, 1, figsize=(12,8))

reviews['points'].value_counts().sort_index().plot.bar(ax=axarr[0])
reviews['province'].value_counts().head(20).plot.bar(ax=axarr[1])
axarr #returns an array with the plots
fig, axarr = plt.subplots(2,2, figsize=(12,8))

reviews['points'].value_counts().sort_index().plot.bar(
    ax=axarr[0][0], fontsize=12, color='pink')
axarr[0][0].set_title("Wine Scores", fontsize=18)

reviews['variety'].value_counts().head(20).plot.bar(
    ax=axarr[1][0], fontsize=12, color='red')
axarr[1][0].set_title("Wine Varities", fontsize=18)

reviews['province'].value_counts().head(20).plot.bar(
ax=axarr[1][1], fontsize=12, color='lightblue')
axarr[1][1].set_title("Wine Origins", fontsize=18)

reviews['price'].value_counts().plot.hist(
    ax=axarr[0][1], fontsize=12, color='green')
axarr[0][1].set_title("Wine Prices", fontsize=18)

plt.subplots_adjust(hspace=.3)

import seaborn as sns
sns.despine()
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
fig, axarr = plt.subplots(2, 1, figsize=(8,8))
fig, axarr = plt.subplots(2, 1, figsize=(8,8))

pokemon['Attack'].plot.hist(ax=axarr[0], title='Pokemon Attack Ratings')

pokemon['Defense'].plot.hist(ax=axarr[1], title='Pokemon Defense Ratings')

