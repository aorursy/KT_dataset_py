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
data = pd.read_csv("../input/Iris.csv")
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(10, 10)) 
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax) 
plt.show()
data.head(10)
data.columns
data.PetalWidthCm.plot(kind = 'line', color = 'g',label = 'PetalWidthCm',linewidth=1,grid = True)
data.SepalLengthCm.plot(kind = 'line', color = 'r',label = 'SepalLengthCm',linewidth=1,grid = True)
plt.legend()     # legend = puts label into plot
plt.xlabel('X Axis')              # label = name of label
plt.ylabel('Y Axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='PetalWidthCm', y='SepalLengthCm',alpha = 0.4,color = 'red')
plt.xlabel("PetalWidthCm")              # label = name of label
plt.ylabel("SepalLengthCm")
plt.title("Scatter Plot")            # title = title of plot
data.SepalLengthCm.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()
filtered_data = data['SepalWidthCm'] > 3
filtered_data
data.head()
and_filtered_data = data[(data['SepalWidthCm']>3) & (data['SepalLengthCm']>4)]
and_filtered_data

