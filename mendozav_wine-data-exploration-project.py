# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt

plt.figure()


# Any results you write to the current directory are saved as output.
wine = pd.read_csv('../input/winemag-data_first150k.csv')
country = wine['country'].value_counts()
country1 = country.iloc[0:5]
variety = wine['variety'].value_counts()
variety1 = variety.iloc[0:5]
winery = wine['winery'].value_counts()
winery1 = winery.iloc[0:5]
country1.plot(kind='bar')
winery1.plot(kind='bar')
variety1.plot(kind='bar')
points = wine['points']
best = points == 100
wine[best].shape
best1 = wine[best]
price1 = best1['price'].value_counts()
price1.plot(kind='bar')
type = best1['variety'].value_counts()
type.plot(kind='bar')
country2 = best1['country'].value_counts()
country2.plot(kind='pie')
price = wine['price'].value_counts()
price.head()
df = pd.DataFrame({'points': points,
                       'price': price}).dropna()
df.corr()



