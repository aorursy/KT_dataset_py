# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt

import seaborn as sns  # visualization tool

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
play = pd.read_csv('../input/googleplaystore.csv')
play.info()
play.memory_usage(deep=True)
# Sum of memory usage

play.memory_usage(deep=True).sum()
sorted(play.Genres.unique())
play.head()
play.tail()
play.sample(10)
play.shape
# Checking how many null datas in dataset

play.isnull().sum()
play.columns
play.columns = play.columns.str.replace(" ", "_")

play.head()
play.dtypes
pd.set_option('display.max_colwidth', 1000)
play.head()
play[play["Type"] == "Free"][['App' , 'Category' , 'Reviews' , 'Size', 'Installs', 'Genres']].set_index("Category").head()
play.Category.value_counts()
play.Category.unique()
# Analyzing the Category

play.Category.value_counts().plot(kind='bar')

plt.title("Category counts")

plt.xlabel("Categories")

plt.ylabel("Count")
play.Type.value_counts()
# Analyzing the Type

play.Type.value_counts().plot(kind='bar')

plt.title("Type counts")

plt.xlabel("Types")

plt.ylabel("Count")
# Analyzing the Type as Horizontal

play.Type.value_counts().plot(kind='barh')

plt.title("Type counts")

plt.xlabel("Count")

plt.ylabel("Types")
# Analyzing the Type

play.Type.value_counts().plot(kind='pie')

plt.title("Pie chart of Types")
# Analyzing the Type

play.Type.value_counts().plot(kind='box')

plt.title("Pie chart of Types")

plt.show()
play.Installs.value_counts()
# Analyzing the Installed Apps

play.Installs.value_counts().plot(kind='bar')

plt.title("Installed App counts")

plt.xlabel("How many installed apps")

plt.ylabel("Count")