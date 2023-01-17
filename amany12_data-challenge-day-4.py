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
import pandas as pd
import matplotlib.pyplot as plt

data= pd.read_csv("../input/cereal.csv")
data.head()
# We can use the seaborn package to plot

# import seaborn and alias it as sns
import seaborn as sns

# make a barplot from a column in our dataframe
sns.countplot(data['mfr']).set_title('mfr bar chart')
# Matplotlib is a little bit involved, so we need to do quite a bit of 
# data manipulation before we can make our bar chart

## data preperation

# count how often each pet preference is observed 

Freq=data['mfr'].value_counts()
print(Freq)
# just FYI: this will get us a list of the names
print(list(Freq.index))
# just FYI: this will get us a list of the counts
print(Freq.values)
# get all the name from our frequency plot & save them for later
labels = list(Freq.index)

print(range(len(labels)))

# generate a list of numbers as long as our number of labels
positionsForBars = list(range(len(labels)))
print(positionsForBars)


## actual plotting

# pass the names and counts to the bar function
plt.bar(positionsForBars, Freq.values) # plot our bars
plt.xticks(positionsForBars, labels)
#plt.xticks(positionsForBars, ('Kelloggs','General Mills','Post','Ralston Purina','Quaker Oats','Nabisco','American Home Food ')) # add lables
plt.title("Frequent of mfr")
