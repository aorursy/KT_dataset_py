# Load everything



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # visualizations

import os
# Read data

data_csv = "/kaggle/input/ecommerce-events-history-in-cosmetics-shop/2019-Oct.csv"

raw_data = pd.read_csv(data_csv)



# Get only purchases

only_purchases = raw_data.loc[raw_data.event_type == 'purchase']
raw_data.head()
# With brands only

purchases_with_brands = only_purchases.loc[only_purchases.brand.notnull()]

top_sellers = purchases_with_brands.groupby('brand').brand.agg([len]).sort_values(by='len', ascending=False)

top_sellers.head(20)
raw_data.loc[raw_data.user_session == "26dd6e6e-4dac-4778-8d2c-92e149dab885"].sort_values(by='event_time')
raw_data.loc[raw_data.user_id == 10280338].sort_values(by='event_time')