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
# Read data csv and transpose 

hivData = pd.read_csv("../input/annual_hiv_deaths_number_all_ages.csv", index_col = 0).T
# Show data head

hivData.head()
# Change name of index and columns

hivData.index.names = ['year']

hivData.columns.names = ['country']
# Check again

hivData.head()
# Indexing, access each country series by its name.

hivData['Indonesia']
# replace all the NaN values with Zero's

Data = hivData.fillna(0)

Data
# Descriptive statistics

Data_sum = Data.describe()

Data_sum
# Obtain the number of existing cases per million

from __future__ import division # we need this to have float division without using a cast

Data.apply(lambda x: x/10)
# Plotting

%matplotlib inline



Data[['United Kingdom', 'Spain', 'Colombia']].plot()

plt.xlabel("Year")

plt.ylabel("Size")

plt.show()
# Plotting using boxplot

Data[['United Kingdom', 'Spain', 'Colombia', 'Japan', 'Argentina', 'Australia']].boxplot()
# Grouping data Count per year

sumData = Data.sum(axis=1)

sumData
# Plotting data grouping per year

%matplotlib inline



sumData.plot.bar()
# mean number case per period

DataPeriod = Data.groupby(lambda x: int(x)>1999).mean()

DataPeriod.index = ['1990-2000', '2001-2011']

DataPeriod
# Time series specific operations

DataTimeSeries = Data.copy()

DataTimeSeries.index = pd.to_datetime(Data.index)

DataTimeSeries
# check country has the highest number

Data.apply(pd.Series.idxmax, axis=1) # ambil nilai yang paling tinggi
#  distributions to have an idea of how the countries are distributed in an average year.

Data_mean = Data.mean()

Data_mean.sort_values().plot(kind='bar', figsize=(24,6))