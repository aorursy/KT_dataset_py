import pandas as pd

import numpy as np
train = pd.read_csv("../input/instacart-market-basket-analysis/order_products__train.csv")

prior = pd.read_csv("../input/instacart-market-basket-analysis/order_products__prior.csv")

orders = pd.read_csv("../input/instacart-market-basket-analysis/orders.csv")

products = pd.read_csv("../input/instacart-market-basket-analysis/products.csv")

aisles = pd.read_csv("../input/instacart-market-basket-analysis/aisles.csv")

departments = pd.read_csv("../input/instacart-market-basket-analysis/departments.csv")
for col in orders.columns: 

    msg = 'column: {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (orders[col].isnull().sum() / orders[col].shape[0]))

    print(msg)
orders['order_dow'].value_counts().idxmax()
orders['order_hour_of_day'].value_counts().idxmax()
orders.info('order_id')
order_id = prior.loc[:,["order_id"]]

print(order_id)
order_id_1 = train.loc[:,["order_id"]]

print(order_id_1)
order_id.duplicated()
order_id = order_id.drop_duplicates()
print(order_id)
print(len(order_id))
order_id_1.duplicated()
order_id_1 = order_id_1.drop_duplicates()
print(order_id_1)
print(len(order_id_1))
order_len = len(order_id) + len(order_id_1)

print(order_len)
A = pd.concat([order_id, order_id_1])
A.duplicated()
A = A.drop_duplicates()
print(len(A))
prior = pd.concat([train, prior])
prior.head()
print(len(prior))
print(products[49301:])
prior1 = prior.join(products.set_index('product_id'), on='product_id')
depart_aisle = pd.merge(departments, products, how='left')
print(depart_aisle)
depart_aisle = depart_aisle[['department_id', 'aisle_id']]
depart_aisle.duplicated()
depart_aisle = depart_aisle.drop_duplicates()
print(depart_aisle)
depart_aisle = depart_aisle.sort_values(by=['department_id','aisle_id'])
print(depart_aisle)
from collections import Counter as cc



pro_name = cc(prior1['product_name'])

print(pro_name.most_common()[:10])
#All_orders = pd.merge(prior1, orders, how='left')
#All_orders = All_orders.sort_values(by=['order_id','order_number','add_to_cart_order'])
#All_orders.head(10)
#All_orders.to_csv("All_orders.csv",mode='w')
prior2 = prior1[['product_name','aisle_id']]
prior2.head()
#prior2 = prior2.sort_values(by=['product_name'])
#prior2.head()
print(prior2[prior2['product_name'].isin([])].count())
product_1 = prior2.groupby(['product_name', 'aisle_id'])['product_name'].count().reset_index(name="product_count")
product_1 = pd.DataFrame(product_1)
product_1.head()
product_1 = product_1.sort_values(by=['aisle_id','product_count'],ascending=[True,False])
product_1.head()
aisle_sum = product_1['product_count'].groupby(product_1['aisle_id']).sum().reset_index(name="aisle_sum")
aisle_sum = pd.DataFrame(aisle_sum)
aisle_sum = aisle_sum.sort_values(by=['aisle_sum'], ascending=[False])
aisle_sum.head()
prior1.head()
prior1.columns[::]
depart_count = prior1[['product_name','department_id']]
depart_count = depart_count['product_name'].groupby(depart_count['department_id']).count().reset_index(name="depart_count")
depart_count = depart_count.sort_values(by=['depart_count'],ascending=[False])
depart_count.head(10)