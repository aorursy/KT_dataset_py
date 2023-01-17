import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

print('### Marketing Funnel by Olist ###')
for idx, file in enumerate(os.listdir('../input/marketing-funnel-olist')):
    print(idx, '-', file)
print('\n---------------------------------------------\n')

print('### Brazilian E-Commerce Public Dataset by Olist ###')
for idx, file in enumerate(os.listdir('../input/brazilian-ecommerce')):
    print(idx, '-', file)
# leads dataset
mql = pd.read_csv('../input/marketing-funnel-olist/olist_marketing_qualified_leads_dataset.csv')
mql.head(10)
# closed deals dataset
cd = pd.read_csv('../input/marketing-funnel-olist/olist_closed_deals_dataset.csv')
cd.head(10)
# marketing funnel dataset (NaNs are leads that did not close a deal)
mf = mql.merge(cd, on='mql_id', how='left')
mf.head(10)
# sellers dataset
sellers = pd.read_csv('../input/brazilian-ecommerce/olist_sellers_dataset.csv')
sellers.head(10)
# marketing funnel merged with sellers (this way you get seller location)
mf_sellers = mf.merge(sellers, on='seller_id', how='left')
mf_sellers.head(10)
# order items dataset
items = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')
items.head(10)
# marketing funnel merged with items (this way you get products sold by sellers)
mf_items = mf.merge(items, on='seller_id', how='left')
mf_items.head(10)
