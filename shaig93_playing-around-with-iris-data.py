# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv')

data.info()
data.head()
data.corr()
f, ax = plt.subplots(figsize=(16,16))

sns.heatmap(data.corr(), annot = True, linewidths = 0.5, fmt = '.2f', ax=ax)

plt.show
data.columns
data.SepalLengthCm.plot(kind = 'line', color = 'k', label = 'SepalLengthCm', linewidth = 1, alpha = 1, grid = True, linestyle = '-')

data.SepalWidthCm.plot(kind = 'line', color = 'b', label = 'SepalWidthCm', linewidth = 1, alpha = 1, grid = True, linestyle = '-')

plt.legend(loc = 'upper right')

plt.xlabel = 'x axis'

plt.ylabel = 'y axis'

plt.title = 'Line Plot'

plt.show()
# Scatter plot

data.plot(kind = 'scatter', x = 'SepalLengthCm', y = 'SepalWidthCm', alpha = 0.5, color = 'g')

plt.xlabel('Length Cm')

plt.ylabel('Width Cm')

plt.title('Length Width Plot')            # title = title of plot

data.SepalLengthCm.plot(kind = 'hist', bins = 50, figsize = (15,15))

plt.show
data.SepalLengthCm.plot(kind = 'hist', bins = 50)

plt.clf()
data = pd.read_csv('../input/Iris.csv')
data.columns
series = data['SepalWidthCm']

print(type(series))

data_frame = data[['SepalWidthCm']]

print(type(data_frame))
x = data['SepalWidthCm']>3.5

data[x]
data[np.logical_and(data['SepalWidthCm']>3.5, data['SepalLengthCm']>5.5)]