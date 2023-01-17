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
data = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-09-14.csv") #attaching values to data 

data.head() #Showing the first values
data.info()
data.corr() #Correlation between values
#correlation map

f,ax = plt.subplots(figsize = (19,17))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.3f',ax=ax)

plt.show()
data.columns
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Open.plot(kind = 'line', color = 'r',label = 'Open',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.Close.plot(color = 'g',label = 'High',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend()     # legend = puts label into plot

plt.xlabel('Open')              # label = name of label

plt.ylabel('High')

plt.title('Line Plot of Open & High')            # title = title of plot

plt.show()
data.plot(kind='scatter', x='Open', y='High', alpha = 1,color = 'blue')

plt.xlabel('Open')              # label = name of label

plt.ylabel('High')

plt.title('Open High Scatter Plot')  

plt.show()
# Histogram

# bins = number of bar in figure

data.Open.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
# Histogram

# bins = number of bar in figure

data.High.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.show()
series = data['Open']        # data['Open'] = series

print(type(series))

data_frame = data[['High']]  # data[['High']] = data frame

print(type(data_frame))
x = data['Open']>15000     # There are only lots of values higher than 15000 Dollars

data[x]
data[np.logical_and(data['Open']>19000, data['High']<20000 )] #Values between 19000 and 20000