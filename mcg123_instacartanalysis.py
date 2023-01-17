import numpy as np

import pandas as pd
order_file = pd.read_csv('../input/orders.csv')

order_prod_file = pd.read_csv('../input/order_products__prior.csv')

prod_file = pd.read_csv('../input/products.csv')
order_table = pd.DataFrame(order_file)

order_prod_table = pd.DataFrame(order_prod_file)

product_table = pd.DataFrame(prod_file)

order_table = order_table[order_table.days_since_prior_order > 0.0] #remove NaNs.
#include eval set and product name

order_prod_table=pd.merge(order_prod_table, order_table, on='order_id', how='left')

order_prod_table=pd.merge(order_prod_table, product_table, on='product_id', how='left')

pd.DataFrame.head(order_prod_table)
prior_table = order_prod_table[order_prod_table['eval_set']=='prior']

train_table = order_prod_table[order_prod_table['eval_set']=='train']

test_table = order_prod_table[order_prod_table['eval_set']=='test']
#example: everything user 1 bought

user_table = prior_table[prior_table['user_id']==1]

user_table.sort_values('days_since_prior_order', 0, False)[:20]
total_freq = pd.crosstab(index=order_prod_table['product_id'], columns=['count'])

total_freq[:10]
reordered_table =  order_prod_table[order_prod_table.reordered==1]

reordered_table [:10]
not_reordered =  order_prod_table[order_prod_table.reordered==0]

not_reordered[:10]
#freq = pd.crosstab(index=reordered_table['product_id'], columns=['count'])

#freq_not = pd.crosstab(index=reordered_table['product_id'], columns=['count'])
top_reorders = reordered_table['product_name'].value_counts().reset_index()[:20]

top_reorders.columns=['product_id', 'count']

top_reorders