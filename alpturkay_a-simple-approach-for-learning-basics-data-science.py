# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # for visualization 


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Iris.csv') # import data
d_data = data.drop(["Id"],axis=1) # I drop id columns in dataset
data.info() # Info about data
data.describe() # Statistical info about data
d_data.corr() # Connections between data
#correlation map
f,ax = plt.subplots(figsize=(13, 13))
sns.heatmap(d_data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
setosa = data[data.Species == "Iris-setosa"]
versicolor = data[data.Species == "Iris-versicolor"]
virginica = data[data.Species == "Iris-virginica"]
# scatter plot 

# PetalLengthCm vs PetalWidthCm
plt.scatter(setosa.PetalLengthCm, setosa.PetalWidthCm, color = 'blue', label = 'setosa')
plt.scatter(versicolor.PetalLengthCm, versicolor.PetalWidthCm, color = 'red', label = 'versicolor')
plt.scatter(virginica.PetalLengthCm, virginica.PetalWidthCm, color = 'yellow', label = 'virginica')

plt.legend()
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.title('PetalWidthCm vs PetalLengthCm')
plt.show()

# SepalWidthCm vs SepalLengthCm
plt.scatter(setosa.SepalLengthCm, setosa.SepalWidthCm, color = 'blue', label = 'setosa')
plt.scatter(versicolor.SepalLengthCm, versicolor.SepalWidthCm, color = 'red', label = 'versicolor')
plt.scatter(virginica.SepalLengthCm, virginica.SepalWidthCm, color = 'yellow', label = 'virginica')
plt.legend()
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.title('SepalWidthCm vs SepalLengthCm')
plt.show()


# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.SepalLengthCm.plot(kind = 'line', color = 'y',label = 'SepalLengthCm',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.PetalLengthCm.plot(color = 'r',label = 'PetalLengthCm',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('id')              # label = name of label
plt.ylabel('length')
plt.title('SepalLengthCm vs PetalLengthCm')            # title = title of plot

plt.show()
