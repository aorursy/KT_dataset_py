# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import seaborn as sns

df = pd.read_csv("../input/Data Set for Regression.csv").fillna(0)
X = df[df.columns[:-1]]
X.head()
print(df.count())
# Distribution of Product Categories
fig, ax = plt.subplots(figsize=(50,10), ncols=3, nrows=1)
left   =  0.125  # the left side of the subplots of the figure
right  =  0.9    # the right side of the subplots of the figure
bottom =  0.1    # the bottom of the subplots of the figure
top    =  0.9    # the top of the subplots of the figure
wspace =  .1     # the amount of width reserved for blank space between subplots
hspace =  1.1    # the amount of height reserved for white space between subplots

# This function actually adjusts the sub plots using the above paramters
plt.subplots_adjust(
    left    =  left, 
    bottom  =  bottom, 
    right   =  right, 
    top     =  top, 
    wspace  =  wspace, 
    hspace  =  hspace
)
plt.subplot(1,3,1)
sns.countplot(x="Product_Category_1", data=X, palette="Greens_d", order = X['Product_Category_1'].value_counts().index)
plt.subplot(1,3,2)
sns.countplot(x="Product_Category_2", data=X, palette="Greens_d", order = X['Product_Category_2'].value_counts().index)
plt.subplot(1,3,3)
sns.countplot(x="Product_Category_3", data=X, palette="Greens_d", order = X['Product_Category_3'].value_counts().index)
fig, ax = plt.subplots(figsize=(15,13))
pc_1_2 = pd.crosstab(index = X["Product_Category_2"], columns = X["Product_Category_3"] )
sns.heatmap(pc_1_2, annot=True, fmt="d")
corr = df.corr()
y = df[df.columns[-1]].values
print(y)

sns.set(color_codes=True)
sns.distplot(y);

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
