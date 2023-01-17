# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("../input/winequalityN.csv")
data.head()
data.tail()
data.info()
set(data.type) 
data_red = data[data.type == "red"]
data_red.head()
data_red.info()
data_red.corr()
f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data_red.corr(), annot = True, linewidths = 0.5, fmt = ".1f", ax = ax)
plt.show()
data_red.alcohol.plot(kind = 'line', color = 'r', linewidth = 1, label = 'alcohol', alpha = .7, grid = True, linestyle = ":")

data_red["residual sugar"].plot(kind = 'line', color = 'g', linewidth = 1, label = 'residual sugar', alpha = 0.7, grid = True, linestyle = "-.")
plt.title("Alcohol - Residual Sugar Line Plot")
plt.show()

data_red.plot(kind = "scatter", x = "residual sugar", y = "alcohol", alpha = .5, color = "r")
plt.title("Alcohol - Residual Sugar Scatter Plot")
plt.show()
data_red.alcohol.plot(kind = 'hist', bins = 50, figsize = (12,12))
plt.title("Alcohol Histogram")
plt.show()
