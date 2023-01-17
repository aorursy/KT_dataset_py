# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/sales_train.csv')
test = pd.read_csv('../input/test.csv')
submission = pd.read_csv('../input/sample_submission.csv')
items = pd.read_csv('../input/items.csv')
item_cats = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
print('train:', train.shape, 'test:', test.shape)
train.head()
item_cats.head()
df1 = pd.merge(items, item_cats, on="item_category_id")
df1.head()
train = pd.merge(train, df1, on="item_id")
train = pd.merge(train, shops, on="shop_id")
train.head()

!pip install googletrans
conv_date = pd.DataFrame(train.groupby('date', as_index=False)['item_price'].sum())
conv_date["day"] = conv_date.date.str.extract("([0-9][0-9]).", expand = False)
conv_date["month"] = conv_date.date.str.extract(".([0-9][0-9]).", expand = False)
conv_date["year"] = conv_date.date.str.extract(".([0-9][0-9][0-9][0-9])", expand =False)
conv_date.head()
conv_date["date"] = pd.to_datetime(conv_date[["year","month","day"]])
plt.plot("date", "item_price", data=conv_date.sort_values(by="date"))
plt.xticks(rotation=45)
plt.show()
shops.head()
