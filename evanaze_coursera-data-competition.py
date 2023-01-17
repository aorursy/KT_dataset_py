import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



!pip install googletrans

from googletrans import Translator



import os
# Import the data files

inDir = '../input/'

shops = pd.read_csv(inDir + 'shops.csv')

item_categories = pd.read_csv(inDir + 'item_categories.csv')

sales_train = pd.read_csv(inDir + 'sales_train.csv')

items = pd.read_csv(inDir + 'items.csv')

sample_submission = pd.read_csv(inDir + 'sample_submission.csv') # just for reference

test = pd.read_csv(inDir + 'test.csv')
# Let's merge the data into one dataframe

train = sales_train.merge(items, on='item_id').merge(item_categories, on='item_category_id').merge(shops, on='shop_id')

del items, item_categories, shops, sales_train

train.drop(columns=['item_id','item_category_id'])

train.head()
# Kind of weird, there are about 250 more id's than unique item names. Let's see about item_categories

print(len(train.item_name.unique())); print(train.item_id.max())

# Only one more item_category than id

print(len(train.item_category_name.unique())); print(train.item_category_id.max())
# Let's add the revenue column as suggested

train['revenue'] = train.item_price * train.item_cnt_day

# Let's also add monthly item count, fill in values for shops when they don't show up in a month, translate the item names to english and parse them
translator = Translator()

item_cats = item_categories.item_category_name.unique()

item_names = items.item_name.unique()

shop_names = shops.shop_name.unique()
for i in range(len(item_cats)):

    trans = translator.translate(item_cats[i]).text

    item_categories.loc[i,'item_category_name_eng'] = trans

item_categories = item_categories.drop(columns='item_category_name')
#errCount = []

#for i in range(len(item_names)):

#    try:

#        trans = translator.translate(item_names[i]).text

#    except Exception:

#        errCount.append(i)

#    items.loc[i,'item_name_eng'] = trans
#for i in range(len(shop_names)):

#    trans = translator.translate(shop_names[i]).text

#    shops.loc[i,'shop_name_eng'] = trans