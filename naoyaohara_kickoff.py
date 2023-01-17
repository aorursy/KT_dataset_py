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
#read October data

raw_data = pd.read_csv('/kaggle/input/ecommerce-behavior-data-from-multi-category-store/2019-Oct.csv')
raw_data.head()
# Get only purchases

only_purchases = raw_data.loc[raw_data.event_type == 'purchase']
only_purchases.head()
#Purchased products by brands

purchases_with_brands = only_purchases.loc[only_purchases.brand.notnull()]

top_sellers = purchases_with_brands.groupby('brand').brand.agg([len]).sort_values(by='len', ascending=False)

top_sellers.head(20)
only_purchases.price.describe()