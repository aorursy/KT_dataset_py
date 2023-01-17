# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/movies_metadata.csv')
data.info()

data.corr()

#correlation map

f,ax  = plt.subplots(figsize=(6, 6))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()

data.head(10)

data.columns

# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.budget.astype(float).plot(kind = 'line', color = 'g',label = 'budget',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')



data.revenue.astype(float).plot(color = 'r',label = 'revenue',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = grafige isim etiketi ekle

plt.xlabel('x axis')              # label = x ekseninin ismi

plt.ylabel('y axis')

plt.title('Line Plot')            # title = başlık

plt.show()