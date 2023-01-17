# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/BlackFriday.csv")
data.info()
data.sort_values("Purchase").head()
data.sort_values("Purchase").tail()
len(data.User_ID.unique())
data.Age.unique()
data.describe(include=['object'])
data.describe()
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()
with sns.axes_style(style=None):
    sns.violinplot("Age", "Purchase", hue="Gender", data=data,
                   split=True, inner="quartile",
                   palette=["lightblue", "lightpink"]);
data["Gender_Age"] = data.Gender + " / " + data.Age
data.head()
sns.boxplot(x=data["Gender_Age"], y=data["Purchase"])
sns.set(rc={'figure.figsize':(15,5)})
plt.show()
grid = sns.FacetGrid(data, row="Gender", col="Age", margin_titles=True)
grid.map(plt.hist, "Purchase", color="pink", density=True);
plt.show()
with sns.axes_style(style='ticks'):
    g = sns.factorplot("Age", "Purchase", "Gender", data=data, height=5, aspect=2)
    g.set_axis_labels("Age", "Purchase");