# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/best-artworks-of-all-time/artists.csv')
data.info()
data.head()
# correlation map

f,ax = plt.subplots(figsize = (10,10))

sns.heatmap(data.corr(),annot = True,linewidths =.5,fmt = '.1f',ax = ax)

plt.show()
data.columns
# Paintings Line Plot

data.paintings.plot(figsize = (10,10),kind = 'line',color = 'red',label = 'paintings',grid = True,alpha = 0.5)

plt.legend(loc = 'upper right')

plt.title('Line Plot')

plt.xlabel('id')

plt.ylabel('paintings')

plt.show()
# id paintings Scatter Plot

data.plot(figsize = (10,10),kind = 'scatter',x = 'id',y = 'paintings',alpha = 0.5,color = 'red')

plt.xlabel('id')

plt.ylabel('paintings')

plt.title('Scatter Plot')

plt.show()
# Histogram

data.paintings.plot(kind = 'hist',bins = 50,grid = True,figsize = (10,10))

plt.title('Histogram')

plt.show()
# Italian Artists

filter = data.nationality == 'Italian'

italians = data[filter]

italians