# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.dtypes
test = pd.read_csv("../input/test.csv")
train.head()
test.head(10)
#Function to return column name based on Index
train.columns.get_values()[0]
#To return column index based on column name
def column_index(train, query_cols):
    cols = train.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols,query_cols,sorter=sidx)]
column_index(train,"card_id")
#Shortcut way to know the column names whenever required
list(train.columns.values)
s1 = train.feature_1.value_counts()

s1


sns.distplot(s1)
s2 = train.first_active_month.value_counts()
s2.head()
def strToDate(string):
    d = datetime.strptime(string, '%Y-%m-%d')
    return d;

 #feature_1 vs target
# Initialize a grid of plots with an Axes for each walk
grid = sns.FacetGrid(train, col="feature_1", hue="feature_1", palette="tab20c",
                     col_wrap=4, height=1.5)

# Draw a horizontal line to show the starting point
grid.map(plt.axhline, y=0, ls=":", c=".5")

# Draw a line plot to show the trajectory of each random walk
grid.map(plt.plot, "target", "first_active_month", marker="o")

# Adjust the tick positions and labels
grid.set(xticks=np.arange(3), yticks=[-3, 3],
         xlim=(-.5, 4.5), ylim=(0, 5))

# Adjust the arrangement of the plots
grid.fig.tight_layout(w_pad=1)
#cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
ax = sns.scatterplot(x="target", y="feature_1",
                     hue="first_active_month", size="feature_1",
                     sizes=(100, 200),
                     data=train)