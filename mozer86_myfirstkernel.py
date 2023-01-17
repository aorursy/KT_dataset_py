# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv")
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(10)
data.columns
# Line Plot
data.Freedom.plot(kind = "line", color = "red",label = "Freedom",linewidth = 1, alpha = 0.6, grid = True,linestyle = ":")
data.Generosity.plot( color = "green",label = "Generosity",linewidth = 1, alpha = 0.5, grid = True,linestyle = "-")
plt.legend(loc = "upper right")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()
#Scatter Plot
data.plot(kind = "scatter",x ="Freedom", y = "Generosity",alpha =.5,color="orange")
plt.xlabel("Freedom")
plt.ylabel("Generosity")
plt.title("Freedom Generosity Scatter Plot")
plt.show()
# Histogram
data.Freedom.plot(kind = "hist", bins = 50,figsize=(10,10))
plt.show()
#clf() = clean it up again
data.Freedom.plot(kind ="hist", bins = 50)
plt.clf()
series = data['Freedom']
print(type(series))
data_frame = data [['Freedom']]
print(type(data_frame))
x= data["Freedom"]< 0.2
data[x]
# Filtering Pandas with logical_and
data[np.logical_and(data["Freedom"]<0.2, data["Happiness.Score"]>5)]
#This line gives us the same result with previous code line.

data[(data["Freedom"]<0.2) & (data["Happiness.Score"]>5)]