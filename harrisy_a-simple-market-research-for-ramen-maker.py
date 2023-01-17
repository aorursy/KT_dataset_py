# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline



data = pd.read_csv('../input/ramen-ratings/ramen-ratings.csv')

data.info()

data
# Distribution of categorical variables brand, style and country (top 5 displayed)

def top_5(var):

    others_sum = data[var].value_counts()[5:].sum()

    s = data[var].value_counts()[:5].append(pd.Series({'Others':others_sum}))

    return s

f, ax = plt.subplots(1, 3, figsize=(20, 7))

col = ['Brand', 'Style', 'Country']

for i in range(len(col)):

    ax[i].pie(top_5(col[i]), labels=top_5(col[i]).index, autopct='%0.2f%%')

    ax[i].set_title('Top 5 {} Distribution'.format(col[i]))
# Distribution of Stars

data = data[data['Stars']!='Unrated']

data['Stars'] = data['Stars'].astype('float32')

sns.distplot(data['Stars'])
# Stars distribution by contries, styles and brands(top 5)

top5contries_list = data['Country'].value_counts()[:5].index

data_top5contries = data[data['Country'].isin(top5contries_list)]

top5brands_list = data['Brand'].value_counts()[:5].index

data_top5brands = data[data['Brand'].isin(top5brands_list)]



f, ax = plt.subplots(3, 1, figsize=(15, 15))

sns.boxplot(x='Country', y='Stars', data=data_top5contries, ax=ax[0], width=0.5)

sns.boxplot(x='Style', y='Stars', data=data, ax=ax[1], width=0.5)

sns.boxplot(x='Brand', y='Stars', data=data_top5brands, ax=ax[2], width=0.5)

for i in range(3):

    ax[i].grid(axis='y')
# Count the frequency of words in 'Variety'(20 most common words displayed)

import collections

v_list = data['Variety'].str.lower().str.split().sum()

collections.Counter(v_list).most_common(20)
# Most Advertised Flavor

data['Flavor'] = data['Variety'].str.lower().str.extract(' ([a-z]+) flavor| ([a-z]+) flavour')

data['Flavor'].value_counts()[:10]
# What kind of chicken is more advertised?

data['Chicken Type'] = data['Variety'].str.lower().str.extract(' ([a-z]+) chicken')

data['Chicken Type'].value_counts()[:10]



# Scpicy chicken or Artificial chicken is more used.
# Ramen Type of 'ramen'

data['Ramen Type'] = data['Variety'].str.lower().str.extract(' ([a-z]+) ramen')

data['Ramen Type'].value_counts()[:10]
# Stars distribution on different ramen types

sns.boxplot(x='Ramen Type', y='Stars', data=data[data['Ramen Type'].isin(['shoyu', 'miso', 'tonkotsu', 'shio'])])



# 'Tonkotsu ramen' seems to have higer ratings than the others overall.
# Does spicy or not affect the ratings by different styles? 

data['Spicy'] = np.where(data['Variety'].str.contains('spicy|chili', case=False), 1, 0)

f, ax = plt.subplots(figsize=(15, 6))

sns.violinplot(x= 'Style', y='Stars', hue='Spicy', data=data, split=True)



# I don't see much differences between spicy or not while no spicy seems slightly better.
# Extract the data with 'Top Ten' values

data_top_10 = data[(data['Top Ten'].notna())&(data['Top Ten']!='\n')]

import re

pattern = re.compile('(\d+)+ #(\d+)')

data_top_10[['Top_10_Year', 'Top_10_No']] = data_top_10['Top Ten'].str.extract(pat=pattern, expand=True)

data_top_10['Top_10_No'] = data_top_10['Top_10_No'].astype('int8')

data_top_10.info()
# Distribution of categorical variables brand, style and country in Top Ten Data

f, ax = plt.subplots(1, 3, figsize=(15, 5))

col = ['Brand', 'Style', 'Country']

for i in range(len(col)):

    counts = data_top_10[col[i]].value_counts()

    ax[i].pie(counts, labels=counts.index, autopct='%0.2f%%')

    ax[i].set_title('{} Distribution in Top 10'.format(col[i]))
# How's the top 10 trend by contries?

data_top_10.sort_values(by='Top_10_Year', inplace=True)

f, ax = plt.subplots(figsize=(15, 10))

sns.lineplot(x='Top_10_Year', y='Top_10_No', hue='Country', ci=None, data=data_top_10)