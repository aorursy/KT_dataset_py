# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
url = "../input/chipotle.tsv"
chipo = pd.read_csv(url, sep = '\t')
chipo.head(10)
chipo.shape[1]
chipo.columns
chipo.index
c = chipo[['item_name', 'quantity']].groupby(['item_name'], as_index = False).agg({'quantity': sum})
c.sort_values(['quantity'], ascending = False, inplace = True)
c.head()
chipo['item_name'].nunique()
chipo['choice_description'].value_counts().head()
total_items_orders = chipo['quantity'].sum()
total_items_orders
dollarizer = lambda x : float(x[1:-1])
chipo['item_price'] = chipo['item_price'].apply(dollarizer)
chipo['sub_total'] = round(chipo['item_price'] * chipo['quantity'], 2)
chipo['sub_total'].sum()
chipo['order_id'].nunique()
chipo[['order_id', 'sub_total']].groupby(['order_id']).agg({'sub_total': sum})['sub_total'].mean()
chipo['item_name'].nunique()