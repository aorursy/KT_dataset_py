# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from seaborn import *



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/municipalities-of-the-netherlands/municipalities_v7.csv')



data.info()

                
data.corr()
f,ax = plt.subplots(figsize=(16, 16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.head(10)
data.columns
data.population.plot(kind = 'line', color = 'g',label = 'Population',linewidth=1,alpha = 0.8,grid = True,linestyle = ':')

data.avg_household_income.plot(color = 'r',label = 'avg_household_income',linewidth=1, alpha = 0.8,grid = True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis') 

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
data.plot(kind='scatter', x='population', y='avg_household_income',alpha = 0.5,color = 'red')

plt.xlabel('population')              # label = name of label

plt.ylabel('avg_household_income')

plt.title('Populatin -Average Household Income Scatter Plot')            # title = title of plot

plt.show()
# Histogram

# bins = number of bar in figure

data.population.plot(kind = 'hist',bins = 70,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.population.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()
data = pd.read_csv('/kaggle/input/municipalities-of-the-netherlands/municipalities_v7.csv')



series = data['population']        # data['Defense'] = series

print(type(series))

data_frame = data[['avg_household_income']]  # data[['Defense']] = data frame

print(type(data_frame))
 #2 - Filtering pandas with logical_and

# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100

data[np.logical_and(data['population']>100000, data['year']>2017 )]
data[(data['population']>800000) & (data['year']>2017)]
plotPerColumnDistribution(data, 10, 5)
