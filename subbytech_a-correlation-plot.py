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
from string import ascii_letters 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



sns.set(style="white")
train_dataset = pd.read_csv("../input/train.csv", parse_dates = ['timestamp'])

macro_dataset = pd.read_csv("../input/macro.csv", parse_dates = ['timestamp'])
train_dataset.head()
final_df = pd.merge(train_dataset, macro_dataset, on='timestamp')
final_df.head()
final_df.fillna(0, inplace=True)
final_df.shape
# Compute the correlation matrix

corr = final_df.corr()
corr
sns.heatmap(corr, 

        xticklabels=corr.columns.values.tolist(),

        yticklabels=corr.columns.values.tolist())
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True
%matplotlib inline
# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3,

            square=True, xticklabels=5, yticklabels=5,

            linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)