# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
etsy_data = pd.read_csv('../input/etsy-shops/etsy_shops_data.csv')

etsy_data.head()
etsy_data['shop_location'] = etsy_data['shop_location'].replace('None', np.nan)

etsy_data['sales_count'] = etsy_data['sales_count'].replace(-99, np.nan)

etsy_data['review_count'] = etsy_data['review_count'].replace(-99, np.nan)

msno.matrix(etsy_data)
corr = etsy_data.corr()

plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')

corrMat = plt.matshow(corr, fignum = 1)

plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)

plt.yticks(range(len(corr.columns)), corr.columns)

plt.gca().xaxis.tick_bottom()

plt.colorbar(corrMat)

plt.title(f'Correlation Matrix for {filename}', fontsize=15)

plt.show()
#scatter_matrix(etsy_data.loc[0:1000,['listing_active_count', 'num_favorers', 'sales_count', 'review_count']], alpha=0.75, figsize=(10, 10), diagonal='kde')

sns.pairplot(etsy_data[0:10000], vars=['listing_active_count', 'num_favorers', 'sales_count', 'review_count'], hue="is_shop_us_based", diag_kind= 'kde', kind="reg")
# Distribution of listings count

print(etsy_data.listing_active_count.describe())

plt.figure(figsize=(10,6))

plt.hist(etsy_data.listing_active_count)

plt.xlabel('number of active listings')
plt.figure(figsize=(10,6))

plt.hist(etsy_data.listing_active_count, range=(0,50))

plt.xlabel('number of active listings')
etsy_data[['listing_active_count', 'num_favorers', 'sales_count', 'review_count']].corr()
plt.figure(figsize=(10,6))



sns.regplot(etsy_data.listing_active_count, etsy_data.sales_count)
# find the shops who have more than 500 sales

etsy_data.loc[etsy_data.sales_count>500]
filtered_data = etsy_data.loc[(etsy_data.listing_active_count<100) & (etsy_data.sales_count<100)]

plt.figure(figsize=(10,6))

sns.regplot(filtered_data.listing_active_count, filtered_data.sales_count)
plt.figure(figsize=(10,6))

sns.lmplot('num_favorers', 'sales_count', hue='sale_message', data=etsy_data, height=5, aspect=1.5)
plt.figure(figsize=(10,6))

sns.swarmplot(etsy_data.loc[etsy_data.sales_count>5].sale_message, etsy_data.loc[etsy_data.sales_count>5].sales_count)
plt.figure(figsize=(10,6))

sns.swarmplot(etsy_data.loc[etsy_data.sales_count>30].is_shop_us_based, etsy_data.loc[etsy_data.sales_count>30].sales_count)