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
data = pd.read_csv("../input/pokemon.csv")
data.info()
#Correlation Map

f,ax = plt.subplots(figsize=(9,9))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)

plt.show()            
data.head(10)
data.columns
data.Speed.plot(kind = "line", color = "red",label = "Speed", linewidth = 1, alpha = 0.7, grid = True)

data.Defense.plot(kind = "line", color = "blue", label = "Defense", linewidth = 1, alpha = 0.5, linestyle = ":")

plt.legend(loc = "upper right")

plt.xlabel("X axis")

plt.ylabel("Y axis")

plt.title("Line plot")

plt.show()