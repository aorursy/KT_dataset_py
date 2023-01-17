# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns  # visualization tool

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/vgsales.csv')

data.head()
data.describe()
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(14, 14))

sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt= '.1f',ax=ax)

plt.show()
data.columns
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.NA_Sales.plot(kind = 'line', color = 'g',label = 'NA_Sales',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.EU_Sales.plot(color = 'r',label = 'EU_Sales',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='NA_Sales', y='EU_Sales',alpha = 0.5,color = 'red')

plt.xlabel('NA_Sales')              # label = name of label

plt.ylabel('EU_Sales')

plt.grid()

plt.title('Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.NA_Sales.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.grid()

plt.show()
# Histogram

# bins = number of bar in figure

data.EU_Sales.plot(kind = 'hist',bins = 50,figsize = (12,12))

plt.grid()

plt.show()
# clf() = cleans it up again you can start a fresh

data.NA_Sales.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
filter1 = data.NA_Sales > 2.5

filter2 = data.EU_Sales > 2.5



data[filter1]
data[np.logical_and(filter1, filter2)]
data[filter1 & filter2]
df1 = data.loc[:3,"NA_Sales"]

df1
eu_sales_num = sum(data["EU_Sales"])/len(data["EU_Sales"])

print(eu_sales_num)

data["EU_Sales_Number"] = ["Yuksek" if i > eu_sales_num else "Dusuk" for i in data.EU_Sales]

data.iloc[:,-1]
data["Publisher"].value_counts(dropna=False)
data.info()
data1 = data.loc[0:1,"EU_Sales"]

data2 = data.loc[0:1,"NA_Sales"]



conc_data_col = pd.concat([data1, data2], axis = 1, names=['eu_sales', 'na_sales'])

conc_data_col

conc_data_col.boxplot(column= 'EU_Sales' ,by = 'EU_Sales')

plt.show()
data_new = data.head(5)

melted = pd.melt(frame=data_new,id_vars = 'Rank', value_vars= ['Genre','Publisher'])

melted
melted.pivot(index = 'Rank', columns = 'variable',values='value')
data.info()
data.EU_Sales.value_counts(dropna = False)