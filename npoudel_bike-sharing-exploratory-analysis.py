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
data1 = pd.read_csv('../input/bike-sharing-dataset/day.csv')
data1.head()
data2 = pd.read_csv('../input/bike-sharing-dataset/hour.csv')
data2.head()
data1.info()
data1[['dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit']] = data1[['dteday', 'season', 'yr', 'mnth', 'holiday', 'weekday',
       'workingday', 'weathersit']].astype('category')

data1.describe()
data2.info()
data2[['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit']] = data2[['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday',
       'workingday', 'weathersit']].astype('category')

data2.describe()
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sns
plt.rcParams["figure.figsize"] =(15,10)
sns.stripplot(x="mnth", y="cnt", data=data1)
plt.show()
sns.swarmplot(x="mnth", y="cnt", hue = 'workingday', data=data1)
plt.show()
sns.swarmplot(x="mnth", y="cnt", hue = 'weekday', data=data1)
plt.show()
sns.swarmplot(x="yr", y="cnt", hue = 'workingday', data=data1)
plt.show()
sns.boxplot(x="yr", y="cnt", hue = 'workingday', data=data1)
plt.show()
sns.boxplot(x="mnth", y="cnt", hue = 'workingday', data=data1)
plt.show()

sns.violinplot(x="mnth", y="cnt", data=data1)
plt.show()
sns.barplot(x="mnth", y="cnt", hue = 'workingday', data=data1)
plt.show()
#f = sns.FacetGrid(data1, row ='mnth' , col = 'weekday', hue = 'mnth')
#f.map(plt.hist, 'cnt')
#plt.show()
sns.boxplot(x="mnth", y="cnt", hue = 'weathersit', data=data1)
plt.show()
sns.stripplot(x="hr", y="cnt", data=data2)
plt.show()
sns.boxplot(x="yr", y="cnt", hue = 'workingday', data=data2)
plt.show()
sns.boxplot(x="hr", y="cnt", data=data2)
plt.show()
sns.boxplot(x="hr", y="cnt", hue = 'workingday', data=data2)
plt.show()
sns.boxplot(x="mnth", y="cnt", hue = 'weathersit', data=data2)
plt.show()