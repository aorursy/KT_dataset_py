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
AppleStore = pd.read_csv('../input/AppleStore.csv')
AppleStore.head()
AppleStore.describe()
AppsGratis = AppleStore.price <= 0
AppsPaga = AppleStore.price > 0
#print(AppleStore[AppsGratis].head())
#print(AppleStore[AppsPaga].head())
AppleStore.price.count()

AppleStoreByPrice = pd.DataFrame({'precio': [ AppleStore[AppsGratis].price.count(), AppleStore[AppsPaga].price.count()]}, index=['Paga', 'Gratis'])
AppleStoreByPrice
plot = AppleStoreByPrice.plot.pie(y='precio', figsize=(10, 10), autopct='%1.1f%%')
#AppleStore.plot(subplots=True, layout=(2, -1), figsize=(6, 6), sharex=False)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 5))
AppleStore[AppleStore.price<=AppleStore.price.describe()['75%']].price.plot.hist(ax=axes[0], bins=10); axes[0].set_title('75%-'); axes[0].set_xlabel('Precio'); axes[0].set_ylabel('# Apps');

AppleStore[AppleStore.price>AppleStore.price.describe()['75%']].price.plot.hist(ax=axes[1], bins=10); axes[1].set_title('75%+'); axes[1].set_xlabel('Precio'); axes[1].set_ylabel('# Apps');


axes = plt.subplots(1, 2, figsize=(20, 10));
#AppleStore[AppleStore.price>AppleStore.price.describe()['75%']].price.plot.hist(subplots=True, layout=(0,2), figsize=(20, 10), sharex=False);
#AppleStore[AppleStore.price>AppleStore.price.describe()['75%']].price.plot.hist(subplots=True, layout=(2,0), figsize=(20, 10), sharex=False);
#AppleStore[AppleStore.price>AppleStore.price.describe()['75%']].price.plot(ax=axes[0][0], bins=12, alpha=0.5, subplots=True)
#AppleStore[AppleStore.price<=AppleStore.price.describe()['75%']].price.plot(bins=12, alpha=0.5, subplots=True)

AppleStore[AppleStore.price<=AppleStore.price.describe()['75%']].price.plot.hist(subplots=True, figsize=(0, 0))
AppleStore[AppleStore.price>AppleStore.price.describe()['75%']].price.plot.hist(subplots=True, figsize=(1, 1))
AppleStorePorcentajes = pd.DataFrame({
                                      '25%': [AppleStore.loc[ AppleStore.price<=AppleStore.price.describe()['25%']].price]
                                     })

'''
                                      '50%': [AppleStore.loc[ (AppleStore.price>AppleStore.price.describe()['25%']) & (AppleStore.price<=AppleStore.price.describe()['50%']) ]],
                                      '75%': [AppleStore.loc[ (AppleStore.price>AppleStore.price.describe()['50%']) & (AppleStore.price<=AppleStore.price.describe()['75%']) ]],
                                      '100%': [AppleStore.loc[AppleStore.price>=AppleStore.price.describe()['75%']]]
'''
AppleStorePorcentajes.head()
hist2 = AppleStore[AppleStore.price<AppleStore.price.describe()['75%']].price.plot.hist(bins=100, alpha=0.5, by='price')
from pandas.plotting import scatter_matrix
scatter_matrix(AppleStore, alpha=0.2, figsize=(6, 6), diagonal='kde')