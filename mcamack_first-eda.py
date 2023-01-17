import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
aisles = pd.read_csv("../input/aisles.csv")

depts = pd.read_csv("../input/departments.csv")

orders = pd.read_csv("../input/orders.csv")

products = pd.read_csv("../input/products.csv")

prior = pd.read_csv("../input/order_products__prior.csv")

train = pd.read_csv("../input/order_products__train.csv")
aisles.head(2)
depts.head(2)
orders.head()
products.head(2)
products = products.merge(aisles, on='aisle_id', how='inner').merge(depts, on='department_id', how='inner');

products.head()
L = [aisles, depts]; del L; #try to free up some memory

train = train.merge(orders, on='order_id', how='left').merge(products, on='product_id', how='left');

train.head()
#Prior is huge and the kernel is having memory issues with such a large merge

#prior.merge(orders, on='order_id', how='left').merge(products, on='product_id', how='left')

L = [orders, products]; del L;

prior.shape
train.shape