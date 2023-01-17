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
All_orders = pd.merge(prior1, orders, how='left')
len(All_orders)
All_orders.head()
All_orders[All_orders['product_name']=='Corn Tortillas']
Gluten = All_orders[All_orders['product_name']=='Corn Tortillas']
Gluten
from collections import Counter as cc



pro_name = cc(Gluten['user_id'])

print(pro_name.most_common()[:20])
from collections import Counter as cc



pro_name = cc(Salsa['user_id'])

print(pro_name.most_common()[:20])
Salsa = All_orders[All_orders['product_name']=='Organic Medium Salsa']
Salsa
A19 = All_orders[All_orders['user_id']==131585]
AA19 = A19['product_name']
AA19.to_csv("AA19.csv",index=False)
AA19 = pd.DataFrame(AA19)
AA19
from collections import Counter as cc



pro_name = cc(AA19['product_name'])

print(pro_name.most_common()[:20])
user1 = All_orders[All_orders['user_id']==132849]

user2 = All_orders[All_orders['user_id']==56639]

user3 = All_orders[All_orders['user_id']==78363]

user4 = All_orders[All_orders['user_id']==204198]

user5 = All_orders[All_orders['user_id']==148932]
user1_pro = user1['product_name']

user1_pro = pd.DataFrame(user1_pro)

user2_pro = user2['product_name']

user2_pro = pd.DataFrame(user2_pro)

user3_pro = user3['product_name']

user3_pro = pd.DataFrame(user3_pro)

user4_pro = user4['product_name']

user4_pro = pd.DataFrame(user4_pro)

user5_pro = user5['product_name']

user5_pro = pd.DataFrame(user5_pro)
print(len(user1_pro), len(user2_pro), len(user3_pro), len(user4_pro), len(user5_pro))

USER = pd.concat([user1_pro, user2_pro, user3_pro, user4_pro, user5_pro])

USER
from collections import Counter as cc



pro_name = cc(USER['product_name'])

A = pro_name.most_common()[:100]

print(A)
A = pd.DataFrame(A)
user3_pro[0:10]
A[21:30]
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