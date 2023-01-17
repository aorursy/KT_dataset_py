# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/tmdb_5000_movies.csv")
data.info()
data.corr()
f, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot = True, linewidths=.5, fmt= ".1f",ax=ax)
data.head()
data.columns
data.vote_average.plot(kind = "line", color = "green", label = "vote_average", linewidth=1, alpha=1, grid = True, linestyle = ":")
data.vote_count.plot(color="red", label="vote_count", linewidth=1, alpha = 0.5 , grid=True, linestyle = "-.")
plt.legend(loc ="upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()
data.plot(kind ="scatter", x= "vote_average", y= "vote_count", alpha= 0.5, color = "red")
plt.xlabel("vote_average")
plt.ylabel("vote_count")
plt.title("vote_average vote_count Scatter Plot")
data.vote_average.plot(kind = "hist", bins = 10, figsize = (15,15))
plt.show
series = data["vote_average"]
print(type(series))
data_frame = data[["revenue"]]
print(type(data_frame))
x = data["vote_count"]>12000
data[x]
data[np.logical_and(data["vote_count"]>1500, data["vote_average"]>8)]
data[(data["vote_count"]>1500) & (data["vote_average"]>8)]
for index, value in data[["vote_count"]][0:1].iterrows():
    print(index, " : ", value)