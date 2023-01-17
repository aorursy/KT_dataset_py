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
data = pd.read_csv('/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv',header=0)
data.head(10)
data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]
data.columns
data.describe()
data.info()
data.corr()
#correlation map
#it helps to feature selection
f,ax = plt.subplots(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True, linewidths=.6, fmt= '.2f',ax=ax)
plt.show()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.quality.plot(kind = 'line', color = 'g',label = 'quality',linewidth=1,alpha = 0.6,grid = True,linestyle = ':', figsize=(8, 8))
data.alcohol.plot(color = 'r',label = 'alcohol',linewidth=1, alpha = 0.6,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
# Scatter Plot 
# x = quality, y = fixed_acidity
data.plot(kind='scatter', x='quality', y='fixed_acidity',alpha = 0.3,color = 'green', figsize=(9, 9))
plt.xlabel('quality')              # label = name of label
plt.ylabel('fixed_acidity')
plt.title('Quality Fixed Acidity Scatter Plot')  
plt.show()
# Histogram
# bins = number of bar in figure
data.alcohol.plot(kind = 'hist',bins = 30,figsize = (10,10), color = 'orange')
plt.show()
# clf() = cleans it up again you can start a fresh
#data.alcohol.plot(kind = 'hist',bins = 50)
#plt.clf()
# We cannot see plot due to clf()
# 1 - Filtering Pandas data frame
x = data['alcohol']>13.5  
data[x]
data[np.logical_and(data['alcohol']>13.5, data['residual_sugar']<1.8 )]
# For pandas we can achieve index and value
for index,value in data[['alcohol']][0:2].iterrows():
    print(index," : ",value)
