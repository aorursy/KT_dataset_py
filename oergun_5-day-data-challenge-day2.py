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

nutritions = pd.read_csv("../input/starbucks-menu-nutrition-drinks.csv")
nutritions.describe()
calories = nutritions["Calories"]
plt.hist(calories, edgecolor = "black")
plt.title("Calories in Starbucks drinks")
plt.xlabel("calories")
plt.ylabel("drinks count")
