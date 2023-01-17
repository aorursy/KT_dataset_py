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
data = pd.read_csv("/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv")
import matplotlib.pyplot as plt

plt.scatter(data.revenue,data.popularity,color = "g", alpha = 0.4)

plt.title("Revenue - Popularity Ratio ")

import seaborn as sns

f,ax = plt.subplots(figsize = (15,15))

sns.heatmap(data.corr(), annot = True, linewidths=5,ax=ax)

plt.show()
va = sum(data.vote_average)/len(data.vote_average)

data["Evaluation"] = ["Excellent" if i > va else "Not Bad" for i in data.vote_average]

#print(data["Evaluation"].head(10),data["vote_count"].head(10),data["original_title"].head(10))

Conclusion = list(zip(data["Evaluation"].head(100),data["original_title"].head(100)))

print(Conclusion)
Conclusions = [x for x in Conclusion]

Conclusions
data.describe()
import pandas as pd

budgets = data.budget.values

revenues = data.revenue.values

data["cost"] = [revenues[i]-budgets[i] for i in range(len(revenues))]

br = pd.concat([data["budget"].head(),data["revenue"].head(),data["cost"].head()],axis=1)

br
data["homepage"].value_counts(dropna = False)
"""data.boxplot(column = "revenue", by = "original_title")

plt.show()"""
data.columns