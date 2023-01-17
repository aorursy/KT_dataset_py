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
data = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

data.info()
data.corr()
f,ax = plt.subplots(figsize=(16,16))

sns.heatmap(data.corr(), annot=True, linewidth=5, fmt= ".1f", ax = ax)

plt.show()
data.head(7)
data.columns
data.EU_Sales.plot(kind = 'line', color = 'y',label = 'EU_Sales',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.NA_Sales.plot(color = 'r',label = 'NA_Sales',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     

plt.xlabel('x axis')              

plt.ylabel('y axis')

plt.title('Line Plot')            

plt.show()
data.plot(kind='scatter', x='Year', y='Global_Sales',alpha = 0.5,color = 'red')

plt.xlabel('Year')              

plt.ylabel('Global_Sales')

plt.title('Year Global-Sales Scatter Plot')            

plt.show()
data.Year.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
data.NA_Sales.plot(kind = 'hist',bins = 50)

plt.clf()
x = data["Year"] < 1985

#print(x)

data[x]
data[np.logical_and(data['Year']>2005, data['NA_Sales']>10 )]
data[(data['Year']>2005) & (data['NA_Sales']>10)]