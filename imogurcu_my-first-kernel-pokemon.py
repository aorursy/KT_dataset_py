# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/pokemon/Pokemon.csv')
data.info()
f,ax = plt.subplots(figsize=(18,18))

sn.heatmap(data.corr(),annot=True, linewidths=.5, fmt='.1f', ax=ax)

plt.show()
data.head(10)
data.columns
# Line Plot

data.HP.plot(kind = 'line', color = 'g', label = 'HP', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':')

data.Defense.plot(color = 'r', label = 'Defense', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.')

plt.legend(loc = 'upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# Scatter Plot

data.plot(kind = 'scatter', x = 'HP', y = 'Defense', alpha = 0.5, color = 'red')

plt.xlabel('HP')

plt.ylabel('Defense')

plt.title('Scatter Plot')

plt.show()
# Histogram

data.Attack.plot(kind = 'hist', bins = 50, figsize = (12,12))

plt.show()
x = data['Attack'] > 170

data[x]
data[np.logical_and(data['Attack'] > 170, data['Defense'] > 110)]
data[(data['Attack'] > 150) & (data['Defense'] > 110)]