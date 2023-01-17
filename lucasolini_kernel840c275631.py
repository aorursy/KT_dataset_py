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
df = pd.read_csv('/kaggle/input/superstore-data/superstore_dataset2011-2015.csv', encoding='latin1')
df
df.columns
df[['Order Date', 'Ship Date', 'Ship Mode', 'Segment', 'Country', 'Market', 'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Order Priority']]
df_selected_cols = df[['Order Date', 'Ship Date', 'Ship Mode', 'Segment', 'Country', 'Market', 'Region', 'Category', 'Sub-Category', 'Sales', 'Quantity', 'Discount', 'Profit', 'Shipping Cost', 'Order Priority']]
df_selected_cols.groupby(['Country', 'Category']).agg({'Sales': 'sum', 'Profit': 'sum'})
import matplotlib.pyplot as plt
df_group_by_country_cat = df_selected_cols.groupby(['Country', 'Category'], as_index=False).agg({'Sales': 'sum', 'Profit': 'sum'})
df_group_by_country_cat[['Country', 'Sales', 'Profit']]
df_group_by_country_cat.groupby('Country').agg({'Sales': 'sum', 'Profit': 'sum'}).sort_values('Sales', ascending=False)
df_group_by_country = df_group_by_country_cat.groupby('Country', as_index=False).agg({'Sales': 'sum', 'Profit': 'sum'}).sort_values('Sales', ascending=False)
fig1, ax1 = plt.subplots()

ax1.pie(df_group_by_country['Sales'], labels=df_group_by_country['Country'], autopct='%1.1f%%', startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



fig1.set_size_inches(15, 15)



plt.show()
df_selected_cols['Segment'].unique()
df_selected_cols[['Market', 'Sales', 'Profit', 'Discount', 'Shipping Cost']]
df_ship_disc = df_selected_cols[['Market', 'Sales', 'Profit', 'Discount', 'Shipping Cost']]
df_ship_disc.groupby('Market').agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'sum', 'Shipping Cost': 'sum'})
df_ship_disc_group_market = df_ship_disc.groupby('Market', as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'sum', 'Shipping Cost': 'sum'})
df_ship_disc_group_market['Sales'] = df_ship_disc_group_market['Sales'].apply(lambda x: "${:.1f}k".format((x/1000)))
df_ship_disc_group_market
df_ship_disc_group_market['Profit'] = df_ship_disc_group_market['Profit'].apply(lambda x: "${:.1f}k".format((x/1000)))
df_ship_disc_group_market
df_ship_disc_group_market['Shipping Cost'] = df_ship_disc_group_market['Shipping Cost'].apply(lambda x: "${:.1f}k".format((x/1000)))
df_ship_disc_group_market
df_ship_disc_group_market['Discount'] = df_ship_disc_group_market['Discount'].apply(lambda x: "${:.2f}k".format((x/1000)))
df_ship_disc_group_market
df_ship_disc_group_market = df_ship_disc.groupby('Market', as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'sum', 'Shipping Cost': 'sum'})
df_ship_disc_group_market['Profit(%)'] = df_ship_disc_group_market['Profit'] / df_ship_disc_group_market['Sales']
df_ship_disc_group_market
df_ship_disc_group_market['Shipping Cost(%)'] = df_ship_disc_group_market['Shipping Cost'] / df_ship_disc_group_market['Sales']
df_ship_disc_group_market
df_ship_disc_group_market['Discount(%)'] = df_ship_disc_group_market['Discount'] / df_ship_disc_group_market['Sales']
df_ship_disc_group_market
df_ship_disc_v2 = df_selected_cols[['Market', 'Ship Mode', 'Sales', 'Profit', 'Discount', 'Shipping Cost']]
df_ship_disc_v2.groupby(['Market', 'Ship Mode']).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'sum', 'Shipping Cost': 'sum'})
df_ship_disc_group_market_shipmode = df_ship_disc_v2.groupby(['Market', 'Ship Mode'], as_index=False).agg({'Sales': 'sum', 'Profit': 'sum', 'Discount': 'sum', 'Shipping Cost': 'sum'})
df_ship_disc_group_market_shipmode
df_ship_disc_group_market_shipmode['Shipping Cost(%)'] = df_ship_disc_group_market_shipmode['Shipping Cost'] / df_ship_disc_group_market_shipmode['Sales']
df_ship_disc_group_market_shipmode
df_ship_disc_group_market_shipmode['Discount(%)'] = df_ship_disc_group_market_shipmode['Discount'] / df_ship_disc_group_market_shipmode['Sales']
df_ship_disc_group_market_shipmode
df_ship_disc_group_market_shipmode['Profit(%)'] = df_ship_disc_group_market_shipmode['Profit'] / df_ship_disc_group_market_shipmode['Sales']
df_ship_disc_group_market_shipmode