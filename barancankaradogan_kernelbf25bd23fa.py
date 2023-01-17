# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # visualization
import seaborn as sns  # visualization

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data_movies = pd.read_csv("../input/tmdb_5000_movies.csv")

data_movies.info()

#correlation map

f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data_movies.corr(), annot=True, linewidth=0.3, fmt=".1f", ax=ax)
plt.show

data_movies.columns

data_movies.budget.plot(kind="line", color="b", label="Budgets", linewidth=0.85, alpha=0.7, grid=True, figsize=(10,10))
data_movies.revenue.plot(kind="line", color="r", label="Revenue", linewidth=0.75, alpha=0.5,)
plt.legend(loc="upper center")
plt.show()

data_movies.plot(kind="Scatter", x="budget", y="revenue", alpha=0.6, grid=True, color="green", figsize=(10,10))
plt.show()
data_movies.revenue.plot(kind="hist", bins=50, figsize=(10,10))
plt.show
filtre = data_movies["vote_average"]>8
data_movies[filtre]
filtre2 = data_movies[(data_movies["vote_average"]>8) & (data_movies["vote_count"]>3000)]
filtre2
best_movies = filtre2

best_movies.info()
best_movies.columns
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(best_movies.corr(), annot=True, linewidth=0.3, fmt=".1f", ax=ax)
plt.show
best_movies.plot(kind="Scatter", x="budget", y="revenue", alpha=0.6, grid=True, color="green", figsize=(10,10), s=150)
plt.show()
best_movies.budget.plot(kind="line", color="b", label="Budgets", linewidth=3, alpha=0.7, grid=True, figsize=(10,10))
best_movies.revenue.plot(kind="line", color="r", label="Revenue", linewidth=3, alpha=0.5,grid=True)
plt.legend(loc="upper center")
plt.show()

