# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
life = pd.read_csv('../input/life.csv')
gdp = pd.read_csv('../input/gdp.xls', encoding='latin1', delimiter='\t')
print(life.columns); print(life.index)
print(gdp.columns); print(gdp.index)
print(np.unique(life.Country)) # or pd.unique(life.Country)
print("Total number of Countries in dataset life: %d" %len(pd.unique(life.Country)))
#print("Total number of Countries in dataset: {}".format(len(pd.unique(life.Country))))
print(np.unique(gdp.Country)) # or pd.unique(life.Country)
print("Total number of Countries in dataset gdp: %d" %len(pd.unique(gdp.Country)))
#print("Total number of Countries in dataset gdp: {}".format(len(pd.unique(gdp.Country))))
pd.unique(life.INEQUALITY)
life = life[life['INEQUALITY'] == 'TOT']
life = life.pivot(index='Country', columns='Indicator', values='Value')
life.head()
print(life.shape)
print(life.columns)
data = pd.merge(gdp, life, how='inner', on='Country')
data.shape
data.head()
data.rename(columns={'2015':'GDP'}, inplace=True)
data.head()
sample = data[['GDP', 'Life satisfaction']].iloc[list(np.random.randint(0,len(data),5))]
#sample.plot(kind='scatter', x='GDP', y='Life satisfaction')
sample = np.float(sample)
