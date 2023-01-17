# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from subprocess import check_output
print (check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon.csv')
data.info()
data.corr()
data.head(20)

# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.Generation.plot(kind = 'line', color = 'g',label = 'Speed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Defense.plot(color = 'r',label = 'Defense',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind = 'scatter',x = 'Speed', y='Attack', alpha =.9, color = 'green')
plt.title('Attack and Speed Analysis')
x = data['Attack'] > 100
y = data['Speed'] > 100
m = data['Sp. Atk'] > 100
n = data['Sp. Def'] > 100
#z=(data['Attack'] > 100) & (data['Speed'] > 100)
z=(x & y & m & n)
data[z]
data.shape
data.info()
print(data['Type 2'].value_counts())
data.describe()
data = pd.read_csv('../input/combats.csv')
data.corr()


