import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
sales_data=pd.read_csv("../input/sales-data-sample.csv")

sales_data.head()
sales_data.describe()
sales_data.dtypes
companies = sales_data[['name','price','date']]

companies.head()
companies_group = companies.groupby('name')

companies_group.size()
total_sales = companies_group.sum()
plot1 = total_sales.plot(kind='bar')
products = sales_data[['product_name','price','date']]

products.head()
products_group = products.groupby('product_name')

products_group.size()
Aggregration
total_sales_by_product = products_group.sum()
plot2 = total_sales_by_product.plot(kind='bar')
plot3 = total_sales.plot(kind='hist')
plot4 = total_sales.plot(kind='box')
plot5 = sales_data.price.plot(kind='box')
plot6 = sales_data.price.plot(kind='kde')