import os

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline



pd.set_option('display.width', 1000)
path = '../input/'



csv_orders = os.path.join(path, 'orders.csv')

csv_order_products_prior = os.path.join(path, 'order_products__prior.csv')

csv_order_products_train = os.path.join(path, 'order_products__train.csv')

csv_products = os.path.join(path, 'products.csv')



df_orders = pd.read_csv(csv_orders, usecols=['order_id', 'eval_set', 'order_hour_of_day'])

df_order_products_prior = pd.read_csv(csv_order_products_prior, usecols=['order_id', 'product_id'])

df_order_products_train = pd.read_csv(csv_order_products_train, usecols=['order_id', 'product_id'])

df_products = pd.read_csv(csv_products, usecols=['product_id', 'product_name'])
# remove any rows referring to the test set

df_orders = df_orders[df_orders.eval_set != 'test']



# drop the eval_set column

df_orders = df_orders.drop(['eval_set'], axis=1)



# concatenate the _prior and _train datasets

df_order_products = pd.concat([df_order_products_prior, df_order_products_train])



# expand every order_id with the list of product_ids in that order_id

df = df_orders.merge(df_order_products, on='order_id')

print(df.head())
## Keep only the top 2000 products

top_products = pd.DataFrame({'total_count': df.groupby('product_id').size()}).sort_values('total_count', ascending=False).reset_index()[:2000]

top_products = top_products.merge(df_products, on='product_id')

print(top_products.head())
# keep only observations that have products in top_products

df = df.loc[df['product_id'].isin(top_products.product_id)]
product_orders_by_hour = pd.DataFrame({'count': df.groupby(['product_id', 'order_hour_of_day']).size()}).reset_index()

product_orders_by_hour['pct'] = product_orders_by_hour.groupby('product_id')['count'].apply(lambda x: x/x.sum()*100)

print(product_orders_by_hour.head(24))
mean_hour = pd.DataFrame({'mean_hour': product_orders_by_hour.groupby('product_id').apply(lambda x: sum(x['order_hour_of_day'] * x['count'])/sum(x['count']))}).reset_index()

print(mean_hour.head())
morning = mean_hour.sort_values('mean_hour')[:25]

morning = morning.merge(df_products, on='product_id')

print(morning.head())
afternoon = mean_hour.sort_values('mean_hour', ascending=False)[:25]

afternoon = afternoon.merge(df_products, on='product_id')

print(afternoon.head())
morning_pct = product_orders_by_hour.merge(morning, on='product_id').sort_values(['mean_hour', 'order_hour_of_day'])

afternoon_pct = product_orders_by_hour.merge(afternoon, on='product_id').sort_values(['mean_hour', 'order_hour_of_day'], ascending=False)
# get list of morning and afteroon product names

morning_product_names = list(morning_pct['product_name'].unique())

morning_product_names = '\n'.join(morning_product_names)

afternoon_product_names = list(afternoon_pct['product_name'].unique())

afternoon_product_names = '\n'.join(afternoon_product_names)



# hack to remove 'Variety Pack' from Orange & Lemon Flavor Variety Pack Sparkling Fruit Beverage

morning_product_names = morning_product_names.replace('Variety Pack ', '')
# Figure Size

fig, ax = plt.subplots(figsize=(12, 8))



# Plot

morning_pct.groupby('product_id').plot(x='order_hour_of_day', 

                                       y='pct', 

                                       ax=ax, 

                                       legend=False,

                                       alpha=0.2,

                                       aa=True,

                                       color='darkgreen',

                                       linewidth=1.5,)

afternoon_pct.groupby('product_id').plot(x='order_hour_of_day', 

                                         y='pct', 

                                         ax=ax, 

                                         legend=False,

                                         alpha=0.2,

                                         aa=True,

                                         color='red',

                                         linewidth=1.5,)



# Aesthetics

# Margins

plt.margins(x=0.5, y=0.05)



# Hide spines

for spine in ax.spines.values():

    spine.set_visible(False)



# Labels

label_font_size = 14

plt.xlabel('Hour of Day Ordered', fontsize=label_font_size)

plt.ylabel('Percent of Orders by Product', fontsize=label_font_size)



# Tick Range

tick_font_size = 10

ax.tick_params(labelsize=tick_font_size)

plt.xticks(range(0, 25, 2))

plt.yticks(range(0, 16, 5))

plt.xlim([-2, 28])



# Vertical line at noon

plt.vlines(x=12, ymin=0, ymax=15, alpha=0.5, color='gray', linestyle='dashed', linewidth=1.0)



# Text

text_font_size = 8

ax.text(0.01, 0.95, morning_product_names,

        verticalalignment='top', horizontalalignment='left',

        transform=ax.transAxes,

        color='darkgreen', fontsize=text_font_size)

ax.text(0.99, 0.95, afternoon_product_names,

        verticalalignment='top', horizontalalignment='right',

        transform=ax.transAxes,

        color='darkred', fontsize=text_font_size);