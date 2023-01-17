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



# read in our data
conflicts = pd.read_csv("../input/african-conflicts/african_conflicts.csv",encoding="latin1")
conflicts
# look at only the numeric columns
conflicts.describe()
# look at all columns, including non-numeric
conflicts.describe(include="all")
# list all the columns
print(conflicts.columns)
# get the fatalities column as a series
interactions = conflicts["INTERACTION"]

# show first five rows
interactions.head()
# plot a histogram of fatalities content
plt.hist(interactions)
plt.title("Interactions in African Conflicts")
plt.hist(interactions, bins=10, edgecolor = "black")
conflicts.hist(column = "INTERACTION", figsize = (12,12))