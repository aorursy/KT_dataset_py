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
nov = pd.read_csv('/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Nov.csv')
# Take only lines with the brand specified

NovWithBrand = nov.dropna(subset=['brand'])
# Group by brand

NovWithBrand = NovWithBrand.groupby('brand')
# Create Brands measures DF

Brands = pd.DataFrame(columns = ['Brand', 'Revenue', 'Count Total', '% Purchase', '% Cart Add', '% Cart Drop', '% Views' , 'Count Products', 'Count Users' ,'AVG Price'])

Brands

i=0

for brand, data in NovWithBrand:

    i+=1

    Brands.loc[i,'Brand'] = brand

    Brands.loc[i,'Revenue'] = data[data['event_type']=='purchase']['price'].sum()

    Brands.loc[i,'Count Total'] = len(data)

    Brands.loc[i,'% Purchase'] = len(data[data['event_type']=='purchase']) / Brands.loc[i,'Count Total']

    Brands.loc[i,'% Cart Add'] = len(data[data['event_type']=='cart']) / Brands.loc[i,'Count Total']

    Brands.loc[i,'% Cart Drop'] = len(data[data['event_type']=='remove_from_cart']) / Brands.loc[i,'Count Total']

    Brands.loc[i,'% Views'] = len(data[data['event_type']=='view']) / Brands.loc[i,'Count Total']

    Brands.loc[i,'Count Products'] = data['product_id'].nunique()

    Brands.loc[i,'Count Users'] = data['user_id'].nunique()

    Brands.loc[i,'AVG Price'] = data[['product_id','price']].drop_duplicates()['price'].mean()
# Print with % style

Brands.style.format({'% Purchase': "{:.2%}",'% Cart Add': "{:.2%}",'% Cart Drop': "{:.2%}",'% Views':"{:.2%}"})
# Top 3 revenue brands (for example)

Brands.sort_values('Revenue', ascending=False).head(3)
# Cast to int and floats data types

Brands[['Revenue','Count Total','Count Products','Count Users']]=Brands[['Revenue','Count Total','Count Products','Count Users']].astype('int')

Brands[['% Purchase','% Cart Add','% Cart Drop','% Views','AVG Price']]=Brands[['% Purchase','% Cart Add','% Cart Drop','% Views','AVG Price']].astype('float')

Brands.info()
# The correlation matrix let's us see which measures are correlated with each other

# For example we might be interested to see which measures are best correlated with revenue

import matplotlib.pyplot as plt

import seaborn as sn

corrMatrix = Brands[['Revenue','Count Total','Count Products','Count Users','% Purchase','% Cart Add','% Cart Drop','% Views','AVG Price']].corr()

fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches

sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)
# Same for category overview

NovWithCategory = nov.dropna(subset=['category_code'])
NovWithCategory = NovWithCategory.groupby('category_code')
Categories = pd.DataFrame(columns = ['Category', 'Revenue', 'Count Total', '% Purchase', '% Cart Add', '% Cart Drop', '% Views' , 'Count Products', 'Count Users' ,'AVG Price'])

i=0

for category, data in NovWithCategory:

    i+=1

    Categories.loc[i,'Category'] = category

    Categories.loc[i,'Revenue'] = data[data['event_type']=='purchase']['price'].sum()

    Categories.loc[i,'Count Total'] = len(data)

    Categories.loc[i,'% Purchase'] = len(data[data['event_type']=='purchase']) / Brands.loc[i,'Count Total']

    Categories.loc[i,'% Cart Add'] = len(data[data['event_type']=='cart']) / Brands.loc[i,'Count Total']

    Categories.loc[i,'% Cart Drop'] = len(data[data['event_type']=='remove_from_cart']) / Brands.loc[i,'Count Total']

    Categories.loc[i,'% Views'] = len(data[data['event_type']=='view']) / Brands.loc[i,'Count Total']

    Categories.loc[i,'Count Products'] = data['product_id'].nunique()

    Categories.loc[i,'Count Users'] = data['user_id'].nunique()

    Categories.loc[i,'AVG Price'] = data[['product_id','price']].drop_duplicates()['price'].mean()
Categories
# Cast to ints and floats data types

Categories[['Revenue','Count Total','Count Products','Count Users']]=Categories[['Revenue','Count Total','Count Products','Count Users']].astype('int')



Categories[['% Purchase','% Cart Add','% Cart Drop','% Views','AVG Price']]=Categories[['% Purchase','% Cart Add','% Cart Drop','% Views','AVG Price']].astype('float')



Categories.info()
Categories[['Revenue','Count Total','Count Products','Count Users','% Purchase','% Cart Add','% Cart Drop','% Views','AVG Price']].corr()
import matplotlib.pyplot as plt

import seaborn as sn

corrMatrix = Categories[['Revenue','Count Total','Count Products','Count Users','% Purchase','% Cart Add','% Cart Drop','% Views','AVG Price']].corr()

fig, ax = plt.subplots(figsize=(10,8))         # Sample figsize in inches

sn.heatmap(corrMatrix, annot=True, linewidths=.5, ax=ax)