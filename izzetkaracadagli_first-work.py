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
data = pd.read_csv("../input/creditcard.csv")



data.info()
data.corr()

plt.show()
#correlation map



f,ax = plt.subplots(figsize = (28, 28))

sns.heatmap(data.corr(), annot=True, linewidths = 5, fmt = ' .1f', ax=ax )

plt.show()



data.head(10)
data.columns
#line plot



data.V24.plot(kind = 'line', color = 'r', label = 'v24', linewidth = 1 , alpha = 1, grid = True, linestyle = '-')

data.V17.plot(kind = 'line', color = 'g', label = 'v17', linewidth = 1 , alpha = 1, grid = True, linestyle = ':')

plt.legend(loc='lower right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('line plot', color = 'b')

plt.show()



#scatter plot



data.plot(kind = 'scatter', x = 'V24', y = 'V17', alpha = 0.5, color = 'black')

plt.xlabel('V24')

plt.ylabel('V17')

plt.title('scattter plot', color = 'b')

plt.show()
# histogram



data.V24.plot(kind = 'hist',bins = 5,figsize = (8,8))

plt.show()
# clf() = cleans it up again you can start a fresh

data.V24.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
series = data['V25']        # data['V25'] = series

print(type(series))

data_frame = data[['V25']]  # data[['Defense']] = data frame

print(type(data_frame))
data = pd.read_csv("../input/creditcard.csv")

# Filtering Pandas data frame

x = data['Time'] < 2

data[x]
# Filtering pandas with logical_and

data[np.logical_and(data['Time']<2,data['V8']<0.23)]

# '&' for filtering



data[(data['Time']<2) & (data['V8']<0.23)]