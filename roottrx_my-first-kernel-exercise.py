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
data = pd.read_csv("../input/tmdb_5000_movies.csv")
data
data.info()
data.columns
data.dtypes #budget = integer, popularity = float
print(data["budget"])

data["popularity"] = data.popularity.astype(int)
data["popularity"] = data.popularity*1000000
print(data["popularity"].dtypes)  
print (data["popularity"])
data.budget.plot(color = "k", label = "Budget", grid = True, linestyle = "--")
data.popularity.plot(color = "r", label = "Popularity", linewidth= 1 , grid = True, linestyle = "--")
plt.show()
plt.scatter(data.budget, data.popularity, color = "r")
data.budget.plot(kind = "hist", bins = 100)
data[np.logical_and(data["vote_average"]>8,data["popularity"]>300000000)]
