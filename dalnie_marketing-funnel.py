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
mql = pd.read_csv("../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv")
mql.head(10)
cdb = pd.read_csv("../input/marketing-funnel-olist/olist_closed_deals_dataset.csv")
mfb = mql.merge(cdb, on='mql_id', how='left')
mfb.head(10)
pd.value_counts(mfb['origin'])
mfb["origin"].value_counts(normalize=True)
mfbi = mql.merge(cdb, on='mql_id', how='inner')
mfbi["origin"].value_counts(normalize=True)
mfbi["landing_page_id"].value_counts(normalize=True)
items = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')
mfb_items = mfb.merge(items, on='seller_id', how='left')
mfb_items.head(10)
origin = mfb_items.groupby(['origin']).sum()
origin.sort_values(by='price', ascending=False)