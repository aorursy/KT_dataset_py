# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True) 

from collections import Counter

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
apps_data  = pd.read_csv('/kaggle/input/shopify-app-store/apps.csv')

apps_categories = pd.read_csv('/kaggle/input/shopify-app-store/categories.csv')

apps_categories_id = pd.read_csv('/kaggle/input/shopify-app-store/apps_categories.csv')
apps_categories.set_index('id', inplace = True)

apps_categories
apps_categories_id
apps_categories
categories_list = []

for index, row in apps_categories_id.iterrows():

    categories_list.append(apps_categories.loc[row['category_id'],'title'])
apps_categories_id['categories'] = categories_list
grouped_cat = apps_categories_id.set_index('app_id').groupby('app_id')['categories'].apply(set)

grouped_cat
apps_data_filtered = apps_data[apps_data.columns[apps_data.columns.isin(['id', 'title', 'developer', 'rating', 'reviews_count', 'pricing_hint'])]]
apps_data_filtered.head()
apps_id = [app_id for app_id in apps_data_filtered['id']]

categories_id = [grouped_cat.loc[app_id] for app_id in apps_id]
apps_data_filtered['category'] = [list(cat) for cat in categories_id]
apps_data_filtered.head()

apps_data_filtered.dropna(axis = 'rows', inplace=True)
apps_data_filtered.isnull().sum()
apps_data_filtered['developer'].describe()
apps_data_filtered['rating'].describe()
apps_data_filtered['pricing_hint'].describe()
apps_data_filtered['category']
Counter(pd.Series(data=categories_list))


category_freq = pd.DataFrame.from_dict(Counter(pd.Series(data=categories_list)), orient='index', columns = ['Frequency',])
category_freq.reset_index()
sns.barplot(x='Frequency', y='index', data=category_freq.reset_index(),)
sns.countplot(y='pricing_hint', data=apps_data_filtered)
apps_data_filtered['reviews_count'].describe()
sns.distplot(apps_data_filtered['rating'])