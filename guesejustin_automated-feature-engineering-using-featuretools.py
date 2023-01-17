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
items = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/items.csv")

items.head()
sales_train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")

sales_train.head()
shops = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/shops.csv")

shops.head()
item_categories = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv")

item_categories.head()
import featuretools as ft# Create new entityset

es = ft.EntitySet(id = 'sales')
es = es.entity_from_dataframe(entity_id = 'shops', dataframe = shops, index = 'shop_id')



es = es.entity_from_dataframe(entity_id = 'item_categories', dataframe = item_categories, index = 'item_category_id')



es = es.entity_from_dataframe(entity_id = 'sales_train', dataframe = sales_train, index = 'id',make_index=True)



es = es.entity_from_dataframe(entity_id = 'items', dataframe = items, index = 'item_id')

# define relations

shopid = ft.Relationship(es['shops']['shop_id'],

                                    es['sales_train']['shop_id'])

# define relations

itid = ft.Relationship(es['items']['item_id'],

                                    es['sales_train']['item_id'],

                     )

es = es.add_relationship(shopid)

es = es.add_relationship(itid)

#itcatid = ft.Relationship(es['items']['item_category_id'],

#                                    es['item_categories']['item_category_id'])
es
# Perform deep feature synthesis without specifying primitives

features, feature_names = ft.dfs(entityset=es, target_entity='sales_train', 

                                 max_depth = 2)



features.head()