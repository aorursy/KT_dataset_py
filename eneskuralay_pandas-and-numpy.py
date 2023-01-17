# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
dataName = "/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv"

data = pd.read_csv(dataName)

data.info()
data.columns
data.head(10)
data.corr()
f,ax = plt.subplots(figsize=(30, 30))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
data.columns

data.price.plot(kind="line",color="orange",label="Price",linewidth=1,alpha=0.8,grid=True,linestyle = ':')

data.minimum_nights.plot(kind="line",color="blue",label="Minimum Nights",linewidth=1,alpha=0.8,grid=True,linestyle = ':')

plt.show()


data.plot(kind='scatter', x='price', y='minimum_nights',alpha = 0.5,color = 'red')

plt.xlabel('Price')              # label = name of label

plt.ylabel('Minimum Nights')

plt.show()
data.price.plot(kind = 'hist',bins = 10,figsize = (6,6))

plt.show()