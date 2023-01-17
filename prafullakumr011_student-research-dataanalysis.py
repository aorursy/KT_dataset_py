# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from pyplot import plot
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
file=pd.ExcelFile("../input/research_student (1).xlsx")
dataset = file.parse('Sheet1', header=0)
dataset.head(5)
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(dataset.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
dataset.columns
dataset.info()
# Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
plt.plot(kind = 'line',color = 'g',label = 'marks[10th]',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
plt.plot(color = 'r',label = 'marks[12]',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
# Scatter Plot 
# x = attack, y = defense
dataset.plot(kind='scatter', x='Marks[10th]', y='Marks[12th]',alpha = 0.5,color = 'red')
plt.xlabel('Marks[10th]')              # label = name of label
plt.ylabel('Marks[12th]')
plt.title('Attack Defense Scatter Plot')            # title = title of plot