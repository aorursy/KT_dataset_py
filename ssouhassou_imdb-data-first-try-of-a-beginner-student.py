# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
movies_data = pd.read_csv("/kaggle/input/imdb-data/IMDB-Movie-Data.csv")

movies_data.head()
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sea
plt.figure(figsize=(10,6))

plt.title("Votes by Years")

sea.barplot(data=movies_data, x="Year", y="Votes")

plt.ylabel("Votes")
plt.figure(figsize=(10,6))

plt.title("Revenue by Years")

sea.barplot(data=movies_data, x="Year", y="Revenue (Millions)")

plt.ylabel("Revenue (Millions)")
plt.figure(figsize=(10,6))

sea.countplot(data=movies_data, x="Year")

plt.ylabel("Number of Movies")
p_grid = sea.PairGrid(movies_data)

p_grid = p_grid.map_diag(plt.hist)

p_grid = p_grid.map_offdiag(plt.scatter)

p_grid = p_grid.add_legend()
fig, ax = plt.subplots(figsize=(8,8))

plt.scatter(data=movies_data,y="Rating", x="Year")

plt.xlabel("Years")

plt.ylabel("Ratings")

plt.title("Ratings of Movies")
movies_ratings = movies_data["Rating"]

fig, ax = plt.subplots(figsize=(8,8))

sea.distplot(movies_ratings, kde=False)
movies_ratings = movies_data["Rating"]

fig, ax = plt.subplots(figsize=(8,8))

sea.distplot(movies_ratings, kde=True)
corr = movies_data.corr()



figure = plt.figure(figsize=(8,8))



sea.heatmap(data=corr, annot=True)
df = movies_data.select_dtypes(exclude="object") 

df.head()
sea.set_style("darkgrid")

sea.lmplot(data= df, y="Votes", x="Revenue (Millions)", height=8)



plt.title("Revenue Vs Votes" )

plt.show()