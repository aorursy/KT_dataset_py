# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/tesla-stock-prices-from-last-decade/TSLA.csv') #data must be imported assigning to a value

data.head(10) #shown first 10 datasets
data.info()
data.corr()
f,ax = plt.subplots(figsize=(15, 15)) #The values inside figsize are calibrating how big the figure will be

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) #By using the correlation dataset, a heat map is constructed using seaborn lib

plt.show()
data.columns #it shows columns of data and as an object
#Line plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.High.plot(kind = 'line', color = 'g',label = 'High price',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Volume.plot(color = 'r',label = 'Volume',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend()     # legend = puts label into plot

plt.xlabel('High price')              # label = name of label

plt.ylabel('Volume')

plt.title('Line Plot')            # title = title of plot

plt.show()
# x = High price, y = Volume

data.plot(kind='scatter', x='High', y='Volume',alpha = 0.5,color = 'red')

plt.xlabel('High price')              # label = name of label

plt.ylabel('Volume')

plt.title('High price Volume Scatter Plot')            # title = title of plot

plt.show()
series = data['High']        # data['High'] = series

print(type(series))

data_frame = data[['Volume']]  # data[['Volume']] = data frame

print(type(data_frame))