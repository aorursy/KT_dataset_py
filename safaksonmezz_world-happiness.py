# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 2015 data

hap15 = pd.read_csv('../input/world-happiness/2015.csv')
hap15.info()
hap15.head()
# Groupby



df1 = hap15.copy()

df1.groupby('Region')['Happiness Score'].mean().sort_values(ascending = False)

df1.groupby('Region')['Happiness Score'].describe()
# Scatter plot

fig1 = plt.figure(figsize = (8,8))

ax1 = fig1.add_axes([0,0,1,1])



sns.scatterplot(x = 'Happiness Score',y = 'Freedom' ,data=df1, hue = 'Region');

ax1.legend(loc=0)
df1.corr()
# Heatmap

sns.heatmap(df1.corr())
# Grids

g = sns.FacetGrid(df1, col= 'Region', hue = 'Region')

g.map(plt.scatter, 'Happiness Score', 'Standard Error')

plt.savefig('countires.jpg',dpi=400)