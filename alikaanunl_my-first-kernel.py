# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data.info()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

data = pd.read_csv('../input/googleplaystore.csv')


data.corr()
print(type(data))
data.head(5)
data.columns
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Rating.plot(kind = 'line', color = 'g',label = 'Rating',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Rating Data')            # title = title of plot
plt.show()
data.Rating.plot(kind = 'hist',bins = 40,figsize = (8,8))
plt.show()
x = data['Rating']>4.5     
data[x]
y = data['Rating']==5
data[y]
data.dtypes