# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Advertising.csv")
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.3, fmt= '.2f',ax=ax)
plt.show()

data.head(10)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.TV.plot(kind = 'line', color = 'g',label = 'TV',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.radio.plot(color = 'r',label = 'Radio',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
data.newspaper.plot(color = 'b',label = 'Newspaper',linewidth=1, alpha = 0.5,grid = True,linestyle = '-')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot TV
# x = TV, y = Sales
data.plot(kind='scatter', x='TV', y='sales',alpha = 0.5,color = 'red')
plt.xlabel('TV')              # label = name of label
plt.ylabel('Sales')
plt.title('TV Sales Scatter Plot')            # title = title of plot
# Scatter Plot TV
# x = Radio, y = Sales
data.plot(kind='scatter', x='radio', y='sales',alpha = 0.5,color = 'green')
plt.xlabel('Radio')              # label = name of label
plt.ylabel('Sales')
plt.title('Radio Sales Scatter Plot')            # title = title of plot
# Scatter Plot Newspaper
# x = newspaper, y = Sales
data.plot(kind='scatter', x='newspaper', y='sales',alpha = 0.5,color = 'blue')
plt.xlabel('Newspaper')              # label = name of label
plt.ylabel('Sales')
plt.title('Newspaper Sales Scatter Plot')            # title = title of plot
# Histogram
# bins = number of bar in figure
data.newspaper.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
# clf() = cleans it up again you can start a fresh
data.newspaper.plot(kind = 'hist',bins = 50)
plt.clf()
# We cannot see plot due to clf()
series = data['TV']        # data['TV'] = series
print(type(series))
data_frame = data[['TV']]  # data[['TV']] = data frame
print(type(data_frame))
# 1 - Filtering Pandas data frame
x = data['sales']>20     # There are only XXX Sales those have higher sale value than 20
print(data[x])
x.count()
data.count()
TV_ort=data.TV.mean()
Radio_ort=data.radio.mean()
News_ort=data.newspaper.mean()
sales_ort=data.sales.mean()
print("TV:",TV_ort, "Radio:", Radio_ort ,"Newspaper:", News_ort, "Sales:", sales_ort)

# 2 - Filtering pandas with logical_and
# 
data[np.logical_and(data['TV']>150, data['sales']>15 )]

# This is also same with previous code line. Therefore we can also use '&' for filtering.
data[(data['TV']>150) & (data['sales']>15)]
# For pandas we can achieve index and value
for index,value in data[['TV']][190:201].iterrows():
    print(index," : ",value)