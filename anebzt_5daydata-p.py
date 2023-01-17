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
# day1: Reading data into a kernel
df = pd.read_csv("../input/cereal.csv")
df.describe()
# day2: Plot a Numeric Variable with a Histogram
import matplotlib.pyplot as plt

# print(df.columns)
col = df["calories"]
plt.hist(col, bins=9, edgecolor = "white")
plt.title("Sugar intake graph")
plt.xlabel("Sugar amount") # label the x axes 
plt.ylabel("Count") # label the y axes
plt.grid()
# day3: Perform a t-test
from scipy.stats import ttest_ind

# get the sodium for hot cerals
hotCereals = df["sodium"][df["type"] == "H"]
# get the sodium for cold ceareals
coldCereals = df["sodium"][df["type"] == "C"]

print(ttest_ind(hotCereals, coldCereals, equal_var=False))

print("Mean sodium for the hot cereals:", hotCereals.mean())
print("Mean sodium for the cold cereals:", coldCereals.mean())

plt.hist(coldCereals, alpha=0.75, label='cold')
plt.hist(hotCereals, label='hot')
plt.legend(loc='upper right')
plt.title("Sodium(mg) content of cereals by type")


# day4: Visualize categorical data with a bar chart

# day5: Using a Chi-Square Test
