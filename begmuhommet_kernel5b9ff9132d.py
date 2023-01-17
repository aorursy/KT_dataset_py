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
data = pd.read_csv("../input/googleplaystore.csv")
data
data.info()
data = pd.read_csv("../input/googleplaystore.csv")
data.head(5)
data.columns
data.corr()
data_rating = data.Rating
data_rating.plot(kind="line", color="r", alpha=0.5, grid=True, linestyle=":")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Rating Plot")
plt.show()
data.plot(kind="scatter", x="Rating", y="Rating", color="g", alpha=0.5, grid=True)
plt.show()
data.Rating.plot(kind="hist", color="b", alpha=0.5, grid=True)
plt.show()
hight_rating = data.Rating > 4.5
data.index.name = "No"
data[hight_rating]