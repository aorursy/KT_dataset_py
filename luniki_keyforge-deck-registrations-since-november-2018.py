# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_hdf("../input/keyforge-deck-registrations.h5")

_ = df.resample("1D").max().plot(figsize=(12,8), title="KeyForge Deck Registrations", legend=False, grid=True)
_ = df.resample("7D").max().diff().plot(figsize=(12,8), title="Registrations per week", legend=False, grid=True, ylim=0)
import matplotlib.pyplot as plt



fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,6))



dfhr = df.resample("1H").max().diff().copy()

dfhr['hour'] = dfhr.index.hour

_ = dfhr.pivot(columns='hour', values="decks").boxplot(ax=axes)

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16,6))

dfdow = df.resample("1D").max().diff().copy()

dfdow['dow'] = dfdow.index.dayofweek

ax = dfdow.pivot(columns='dow', values="decks").boxplot(ax=axes)

_ = ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])