import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))
movies=pd.read_csv("../input/tmdb_5000_movies.csv")

credits=pd.read_csv("../input/tmdb_5000_credits.csv")

movies.info()
movies.describe()
movies.corr()
f,ax=plt.subplots(figsize=(10,10))

sns.heatmap(movies.corr(),annot=True, linewidths=.5, ax=ax)

plt.show()
movies.budget.plot(label="Budget",grid=True,figsize=(12,9),alpha=0.5, color="r",linestyle="-.")

movies.revenue.plot(label="Revenue", alpha=0.5, color="g",linestyle=":")

plt.title("Line Plot - Budget and Revenue")

plt.legend()

plt.xlabel("id")

plt.ylabel("Price")

plt.show()
movies.plot(kind="scatter",x="vote_average", y="runtime", alpha=.5,figsize=(12,9),label="Scatter Plot - Runtime and Vote Average")

plt.show()
movies.vote_average.plot(kind="hist",bins=30 , figsize=(12,9))

plt.xlabel("Average Vote")

plt.show()
movies[(movies["revenue"]>1200000000)&(movies["budget"]<200000000)]