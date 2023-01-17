# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

data = pd.read_csv("../input/20160307hundehalter.csv")
data.head()
dataTable = data["GESCHLECHT"].value_counts()
list(dataTable.index)
dataTable.values

labels = list(dataTable.index)
positionsForBars = list(range(len(labels)))

plt.bar(positionsForBars, dataTable.values) 
plt.xticks(positionsForBars, labels)
sns.countplot(data["GESCHLECHT"])