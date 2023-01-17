import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as pt # plotting 
import seaborn as sns # pretty plotting

%matplotlib inline

orders = pd.read_csv('../input/orders.csv')
products = pd.read_csv('../input/products.csv')
order_products_train = pd.read_csv('../input/order_products__train.csv')
order_products_prior = pd.read_csv('../input/order_products__prior.csv')
aisles = pd.read_csv('../input/aisles.csv')
departments = pd.read_csv('../input/departments.csv')
print(orders.head(5))
